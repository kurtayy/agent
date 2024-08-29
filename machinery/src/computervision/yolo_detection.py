import sys
import json
from ultralytics import YOLO

try:
    # Load a model
    model = YOLO('yolov8n.pt')  # Specify the correct YOLO model

    # Get the image path from command line arguments
    image_path = sys.argv[1]

    # Run detection
    results = model(image_path)

    # Print the results in JSON format
    detections = [{"class": r['class'], "confidence": r['confidence']} for r in results.xywhn[0].cpu().numpy()]
    print(json.dumps(detections))
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(2)
