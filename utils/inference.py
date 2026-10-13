from ultralytics import YOLO
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


# Load a YOLO model from the given path
def load_model(model_path):
    model = YOLO(model_path)
    return model


# Draw bounding boxes and labels on the image for each detected object
def draw_boxes_on_image(image, boxes, model_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        class_id = int(box.cls)  # Get class id
        class_name = model_names[class_id]  # Get class name
        confidence = box.conf[0]  # Get confidence score

        if box.id is not None:
            track_id = int(box.id[0])
            label = f"{class_name}{track_id}: {confidence:.2f}"
        else:
            label = f"{class_name}: {confidence:.2f}"  # Prepare label text
        # Set color based on class id
        color = (0, 0, 255) if class_id == 0 else (255, 255, 0) if class_id == 1 else (0, 255, 0)
        # Draw label at different positions based on class
        if class_id == 0:
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_id == 1:
            cv2.putText(image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_id == 2:
            cv2.putText(image, label, (x1 + 20, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw rectangle for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def visualize_predicted_image(img_bgr, window_title="Image"):
    # Convert BGR→RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(window_title)
    plt.axis('off')  # turn off axes
    plt.show()


# Run inference on a set of test images and optionally save the results
def predict_test_set(model_path, test_images_path, save=False, output_path=None):
    # Load the YOLO model
    model = load_model(model_path)
    # Collect all .png and .jpg images in the test directory
    test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))

    # Run model prediction on the test images
    results = model.predict(source=test_images, batch=1, show=False, show_labels=False, save=False)
    print(results)
    # Iterate through each result and process the images
    for idx, result in enumerate(results):
        # Draw bounding boxes and labels on the original image
        img_with_boxes = draw_boxes_on_image(result.orig_img, result.boxes, model.names)
        # Display the image with bounding boxes
        # cv2.imshow('Image with Bounding Boxes', img_with_boxes)
        # visualize_predicted_image(img_with_boxes)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # If saving is enabled and output path is provided, save the image
        if save and output_path:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            out_file = os.path.join(output_path, f"pred_{idx}.jpg")
            cv2.imwrite(str(out_file), img_with_boxes)

    # Return the results for further processing if needed
    return results

def predict_single_image(model_path: Path, test_images_path: Path, save=False, output_path=None):

    model = load_model(model_path)

    test_images = (list(test_images_path.glob("*.png")) + list(test_images_path.glob("*.jpg")))

    all_results = []
    if save and output_path:
        os.makedirs(output_path, exist_ok=True)

    for image_path in tqdm(test_images):
        results = model.predict(source=image_path, show=False, show_labels=False, save=False, verbose=False)
        # results = model.track(source=image_path, show=False, show_labels=False, save=False, verbose=False)
        result = results[0]
        all_results.append(result)

        img_with_boxes = draw_boxes_on_image(result.orig_img.copy(),
                                             result.boxes,
                                             result.names)

        if save and output_path:
            filename = image_path.stem
            prefix = "_".join(filename.split("_")[:3])

            new_filename = output_path / f"{prefix}_pred.png"
            cv2.imwrite(new_filename, img_with_boxes)

    return all_results



# Run real-time inference using the webcam
def webcam_inference(model_path):
    # Load the YOLO model
    model = load_model(model_path)
    # Open the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit. Press 's' to save the current frame with prediction.")
    save_count = 0
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break  # Exit loop if frame not captured
        # Run prediction on the current webcam frame
        results = model.predict(source=[frame], show=False, show_labels=False, save=False, verbose=False)
        result = results[0]
        # Draw bounding boxes and labels on the frame
        img_with_boxes = draw_boxes_on_image(frame.copy(), result.boxes, model.names)
        # Display the frame with predictions
        cv2.imshow('Webcam Inference', img_with_boxes)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit if 'q' is pressed
        elif key == ord('s'):
            # Save the current frame with predictions if 's' is pressed
            filename = f"webcam_pred_{save_count}.jpg"
            cv2.imwrite(filename, img_with_boxes)
            print(f"Saved: {filename}")
            save_count += 1
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == '__main__':
    # Define the paths
    model_path = Path('C:/Users/panosvrachnos/Documents/thesis_env/YOLO/yolo_v2/yolo_v2_weights/best.pt')
    test_images_path = Path(r'C:\Users\panosvrachnos\Documents\thesis_env\YOLO\yolo_v2\from_top_screen_view_grid_images')
    output_path = Path('C:/Users/panosvrachnos/Desktop/yolo_inference/from_top_views')

    # Predict on the test set
    #results = predict_test_set(model_path, test_images_path, save=True, output_path=output_path)
    results = predict_single_image(model_path, test_images_path, save=True, output_path=output_path)

    # Uncomment the following line to perform webcam inference
    # webcam_inference(model_path)