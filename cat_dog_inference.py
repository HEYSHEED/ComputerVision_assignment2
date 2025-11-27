import joblib
import cv2
import numpy as np

IMG_SIZE = 64
model = joblib.load("cat_dog_best_model.z")
scaler = joblib.load("cat_dog_scaler.z")

def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image:", image_path)
        return

    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = img.flatten().reshape(1, -1)

    img_sc = scaler.transform(img)

    pred = model.predict(img_sc)[0]

    label = "Cat" if pred == 0 else "Dog"

    output = image.copy()
    cv2.putText(output, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(f"Prediction for {image_path} -> {label}")
    cv2.imshow(label, output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_image_path = r"C:\Users\sheed\OneDrive\Desktop\cvi\ComputerVisionFall2025\Assignment-2\image_internet.jpg"
    predict_image(test_image_path)
