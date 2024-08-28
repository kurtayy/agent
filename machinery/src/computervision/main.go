package computervision

import (
	"gocv.io/x/gocv" // Assuming OpenCV for YOLO support
	"image"
	"time"
	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/kerberos-io/agent/machinery/src/capture"
	"github.com/kerberos-io/agent/machinery/src/conditions"
	"github.com/kerberos-io/agent/machinery/src/geo"
	"github.com/kerberos-io/agent/machinery/src/models"
	"github.com/kerberos-io/agent/machinery/src/packets"
	"github.com/kerberos-io/agent/machinery/src/utils"
)

// Initialize the YOLO model
func InitYOLOModel() gocv.Net {
	// Load the YOLO model
	yoloModel := "yolov3.weights" // Path to the YOLO weights file
	yoloConfig := "yolov3.cfg"    // Path to the YOLO config file
	backend := gocv.NetBackendDefault
	target := gocv.NetTargetCPU

	net := gocv.ReadNet(yoloModel, yoloConfig)
	net.SetPreferableBackend(backend)
	net.SetPreferableTarget(target)

	return net
}

// YOLO detection function
func DetectHumanWithYOLO(img gocv.Mat, net gocv.Net) bool {
	// Set up YOLO input size and swap RGB
	blob := gocv.BlobFromImage(img, 1/255.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	// Set the input to the YOLO network
	net.SetInput(blob, "")

	// Run forward pass to get output
	output := net.ForwardLayers([]string{"yolo_82", "yolo_94", "yolo_106"})
	defer func() {
		for i := range output {
			output[i].Close()
		}
	}()

	// Iterate over the detections
	for i := range output {
		data, _ := output[i].DataPtrFloat32()
		for j := 0; j < len(data); j += 7 {
			confidence := data[4]
			if confidence > 0.5 { // Threshold for confidence
				classID := data[5]
				if int(classID) == 0 { // Assuming classID 0 is for "person" in the YOLO model
					return true // Human detected
				}
			}
		}
	}
	return false
}

func ProcessMotion(motionCursor *packets.QueueCursor, configuration *models.Configuration, communication *models.Communication, mqttClient mqtt.Client, rtspClient capture.RTSPClient) {

	log.Log.Debug("computervision.main.ProcessMotion(): start motion detection")
	config := configuration.Config
	loc, _ := time.LoadLocation(config.Timezone)

	var isPixelChangeThresholdReached = false
	var changesToReturn = 0

	pixelThreshold := config.Capture.PixelChangeThreshold
	if pixelThreshold == 0 {
		pixelThreshold = 150
	}

	if config.Capture.Continuous == "true" {
		log.Log.Info("computervision.main.ProcessMotion(): you've enabled continuous recording, so no motion detection required.")
	} else {
		log.Log.Info("computervision.main.ProcessMotion(): motion detection is enabled, starting the motion detection.")

		hubKey := config.HubKey
		deviceKey := config.Key

		// Initialize first 2 elements
		var imageArray [3]*image.Gray
		j := 0

		var cursorError error
		var pkt packets.Packet

		for cursorError == nil {
			pkt, cursorError = motionCursor.ReadPacket()
			if len(pkt.Data) > 0 && pkt.IsKeyFrame {
				grayImage, err := rtspClient.DecodePacketRaw(pkt)
				if err == nil {
					imageArray[j] = &grayImage
					j++
				}
			}
			if j == 3 {
				break
			}
		}

		// Calculate mask
		var polyObjects []geo.Polygon

		if config.Region != nil {
			for _, polygon := range config.Region.Polygon {
				coords := polygon.Coordinates
				poly := geo.Polygon{}
				for _, c := range coords {
					x := c.X
					y := c.Y
					p := geo.NewPoint(x, y)
					if !poly.Contains(p) {
						poly.Add(p)
					}
				}
				polyObjects = append(polyObjects, poly)
			}
		}

		img := imageArray[0]
		var coordinatesToCheck []int
		if img != nil {
			bounds := img.Bounds()
			rows := bounds.Dy()
			cols := bounds.Dx()

			for y := 0; y < rows; y++ {
				for x := 0; x < cols; x++ {
					for _, poly := range polyObjects {
						point := geo.NewPoint(float64(x), float64(y))
						if poly.Contains(point) {
							coordinatesToCheck = append(coordinatesToCheck, y*cols+x)
						}
					}
				}
			}
		}

		// Initialize YOLO model
		net := InitYOLOModel()
		defer net.Close()

		// Start the motion detection
		if len(coordinatesToCheck) > 0 {
			i := 0

			for cursorError == nil {
				pkt, cursorError = motionCursor.ReadPacket()
				if len(pkt.Data) == 0 || !pkt.IsKeyFrame {
					continue
				}

				grayImage, err := rtspClient.DecodePacketRaw(pkt)
				if err == nil {
					imageArray[2] = &grayImage
				}

				// Validate conditions (time window, URI response, etc.)
				detectMotion, err := conditions.Validate(loc, configuration)
				if !detectMotion && err != nil {
					log.Log.Debug("computervision.main.ProcessMotion(): " + err.Error() + ".")
				}

				if config.Capture.Motion != "false" && detectMotion {

					// Check for pixel change to detect motion
					isPixelChangeThresholdReached, changesToReturn = FindMotion(imageArray, coordinatesToCheck, pixelThreshold)
					if isPixelChangeThresholdReached {

						// YOLO-based human detection
						yoloImg := gocv.NewMatFromBytes(imageArray[2].Bounds().Dy(), imageArray[2].Bounds().Dx(), gocv.MatTypeCV8U, imageArray[2].Pix)
						defer yoloImg.Close()
						if DetectHumanWithYOLO(yoloImg, net) {
							log.Log.Info("computervision.main.ProcessMotion(): Human detected with YOLO.")

							// If offline mode is disabled, send a message to the hub
							if config.Offline != "true" {
								if mqttClient != nil {
									if hubKey != "" {
										message := models.Message{
											Payload: models.Payload{
												Action:   "motion",
												DeviceId: configuration.Config.Key,
												Value: map[string]interface{}{
													"timestamp": time.Now().Unix(),
												},
											},
										}
										payload, err := models.PackageMQTTMessage(configuration, message)
										if err == nil {
											mqttClient.Publish("kerberos/hub/"+hubKey, 0, false, payload)
										} else {
											log.Log.Info("computervision.main.ProcessMotion(): failed to package MQTT message: " + err.Error())
										}
									} else {
										mqttClient.Publish("kerberos/agent/"+deviceKey, 2, false, "motion")
									}
								}
							}

							if config.Capture.Recording != "false" {
								dataToPass := models.MotionDataPartial{
									Timestamp:       time.Now().Unix(),
									NumberOfChanges: changesToReturn,
								}
								communication.HandleMotion <- dataToPass // Save data to the channel
							}
						} else {
							log.Log.Info("computervision.main.ProcessMotion(): Motion detected but no human found by YOLO, ignoring.")
						}
					}

					imageArray[0] = imageArray[1]
					imageArray[1] = imageArray[2]
					i++
				}
			}

			if img != nil {
				img = nil
			}
		}
	}

	log.Log.Debug("computervision.main.ProcessMotion(): stop the motion detection.")
}


func FindMotion(imageArray [3]*image.Gray, coordinatesToCheck []int, pixelChangeThreshold int) (thresholdReached bool, changesDetected int) {
	image1 := imageArray[0]
	image2 := imageArray[1]
	image3 := imageArray[2]
	threshold := 60
	changes := AbsDiffBitwiseAndThreshold(image1, image2, image3, threshold, coordinatesToCheck)
	return changes > pixelChangeThreshold, changes
}

func AbsDiffBitwiseAndThreshold(img1 *image.Gray, img2 *image.Gray, img3 *image.Gray, threshold int, coordinatesToCheck []int) int {
	changes := 0
	for i := 0; i < len(coordinatesToCheck); i++ {
		pixel := coordinatesToCheck[i]
		diff := int(img3.Pix[pixel]) - int(img1.Pix[pixel])
		diff2 := int(img3.Pix[pixel]) - int(img2.Pix[pixel])
		if (diff > threshold || diff < -threshold) && (diff2 > threshold || diff2 < -threshold) {
			changes++
		}
	}
	return changes
}
