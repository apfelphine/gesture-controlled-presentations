import os
import matplotlib.pyplot as plt
import tensorflow as tf

from mediapipe_model_maker import gesture_recognizer

dataset_path = "../data/training"

print(f"Dataset path: {dataset_path}")


def get_data():
    data = gesture_recognizer.Dataset.from_folder(
        dirname=dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams(
            shuffle=True,
            min_detection_confidence=0.85
        ),
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)
    return train_data, validation_data, test_data


def train(train_data, validation_data):
    hparams = gesture_recognizer.HParams(export_dir="gesture_recognizer_model")
    hparams.epochs = 50
    hparams.learning_rate = 0.005
    hparams.batch_size = 32

    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data, validation_data=validation_data, options=options
    )
    return model


def main():
    train_data, validation_data, test_data = get_data()
    model = train(train_data, validation_data)
    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss}, Test accuracy: {acc}")
    model.export_model()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
