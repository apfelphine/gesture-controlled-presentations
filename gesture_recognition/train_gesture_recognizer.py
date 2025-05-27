import os
import matplotlib.pyplot as plt
import tensorflow as tf

from mediapipe_model_maker import gesture_recognizer

dataset_path = "../data/training"

print(dataset_path)


def get_labels():
    labels = []
    for i in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, i)):
            labels.append(i)
    return labels


def show_labels():
    print(get_labels())


def show_sample_images():
    NUM_EXAMPLES = 5

    for label in get_labels():
        print(label)
        label_dir = os.path.join(dataset_path, label)
        example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
        fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10, 2))
        for i in range(NUM_EXAMPLES):
            axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        fig.suptitle(f"Showing {NUM_EXAMPLES} examples for {label}")

    plt.show()


def get_data():
    data = gesture_recognizer.Dataset.from_folder(
        dirname=dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams(shuffle=True),
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)
    return train_data, validation_data, test_data


def train(train_data, validation_data):
    hparams = gesture_recognizer.HParams(export_dir="gesture_recognizer_model")
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data, validation_data=validation_data, options=options
    )
    return model


def main():
    train_data, validation_data, test_data = get_data()
    model = train(train_data, validation_data)
    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss:{loss}, Test accuracy:{acc}")
    model.export_model()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
