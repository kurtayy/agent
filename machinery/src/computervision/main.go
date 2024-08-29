package computervision

import (
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"os/exec"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	geo "github.com/kellydunn/golang-geo"
	"github.com/kerberos-io/agent/machinery/src/capture"
	"github.com/kerberos-io/agent/machinery/src/conditions"
	"github.com/kerberos-io/agent/machinery/src/log"
	"github.com/kerberos-io/agent/machinery/src/models"
	"github.com/kerberos-io/agent/machinery/src/packets"
)

// Detection struct for YOLO output
type Detection struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
}

func ProcessMotion(motionCursor *packets.QueueCursor, configuration *models.Configuration, communication *models.Communication, mqttClient mqtt.Client, rtspClient capture.RTSPClient) {

	log.Log.Debug("computervision.main.ProcessMotion(): start motion detection")
	config := configuration.Config
	loc, _ := time.LoadLocation(config.Timezone)

	var isPixelChangeThresholdReached = false
	var changesToReturn = 0

	pixelThreshold := config.Capture.PixelChangeThreshold
	// Might not be set in the config file, so set it to 150
	if pixelThreshold == 0 {
		pixelThreshold = 150
	}

	if config.Capture.Continuous == "true" {

		log.Log.Info("computervision.main.ProcessMotion(): you've enabled continuous recording, so no motion detection required.")

	} else {

		log.Log.Info("computervision.main.ProcessMotion(): motion detected is enabled, so starting the motion detection.")

		hubKey := config.HubKey
		deviceKey := config.Key

		// Initialise first 2 elements
		var imageArray [3]*image.Gray

		j := 0

		var cursorError error
		var pkt packets.Packet

		for cursorError == nil {
			pkt, cursorError = motionCursor.ReadPacket()
			// Check If valid package.
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

			// Make fixed size array of uinty8
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

		// If no region is set, we'll skip the motion detection
		if len(coordinatesToCheck) > 0 {

			// Start the motion detection
			i := 0

			for cursorError == nil {
				pkt, cursorError = motionCursor.ReadPacket()

				// Check If valid package.
				if len(pkt.Data) == 0 || !pkt.IsKeyFrame {
					continue
				}

				grayImage, err := rtspClient.DecodePacketRaw(pkt)
				if err == nil {
					imageArray[2] = &grayImage
				}

				if imageArray[2] == nil {
					log.Log.Error("Image array contains nil. Skipping YOLO detection.")
					continue
				}

				// Validate conditions
				detectMotion, err := conditions.Validate(loc, configuration)
				if !detectMotion && err != nil {
					log.Log.Debug("computervision.main.ProcessMotion(): " + err.Error() + ".")
					continue // Skip the rest of the loop if conditions are not met
				}

				if config.Capture.Motion != "false" && detectMotion {

					// Run YOLO detection after validating the conditions
					tempImagePath := "/tmp/frame.jpg" // Choose a suitable temp path
					err = saveImageToFile(imageArray[2], tempImagePath)
					if err != nil {
						log.Log.Error(fmt.Sprintf("Failed to save image to %s: %s", tempImagePath, err.Error()))
						continue
					}

					detections, err := runYOLODetection(tempImagePath)
					if err != nil {
						log.Log.Error("YOLO detection failed: ")
					} else {
						for _, detection := range detections {
							if detection.Class == "person" && detection.Confidence > 0.5 {
								log.Log.Info("Human detected by YOLO with confidence: ")
								// Handle human detection as needed
							}
						}
					}

					// Perform motion detection
					isPixelChangeThresholdReached, changesToReturn = FindMotion(imageArray, coordinatesToCheck, pixelThreshold)
					if isPixelChangeThresholdReached {

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
					}
				}

				// Shift images for next iteration
				imageArray[0] = imageArray[1]
				imageArray[1] = imageArray[2]
				i++
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

// saveImageToFile saves an image to the specified path.
func saveImageToFile(img *image.Gray, path string) error {
	// Create the file
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Encode the image to the file in JPEG format
	if err := jpeg.Encode(file, img, nil); err != nil {
		return err
	}

	return nil
}

func runYOLODetection(imagePath string) ([]Detection, error) {
	cmd := exec.Command("python3", "yolo_detection.py", imagePath)

	output, err := cmd.CombinedOutput() // Capture both stdout and stderr
	if err != nil {
		log.Log.Error(fmt.Sprintf("Error running YOLO detection script: %s, output: %s", err.Error(), string(output)))
		return nil, err
	}

	if len(output) == 0 {
		log.Log.Error("YOLO detection script returned empty output.")
		return nil, fmt.Errorf("empty output from YOLO detection script")
	}

	var detections []Detection
	if err := json.Unmarshal(output, &detections); err != nil {
		log.Log.Error(fmt.Sprintf("Error parsing YOLO detection output: %s", err.Error()))
		return nil, err
	}

	return detections, nil
}
