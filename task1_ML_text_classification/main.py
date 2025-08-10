import utils
from model import LogisticRegression

# --- 参数配置 ---
MAX_WORDS = 10000
LEARNING_RATE = 0.1
EPOCHS = 50  # 增加 epochs 数量以在小数据集上看到效果
BATCH_SIZE = 2 # 使用更小的 batch size

def main():
    """
    主函数，用于训练和评估模型
    """
    print("1. Loading data from hardcoded source...")
    x_train, y_train, x_dev, y_dev, x_test, y_test, vocab = utils.load_rotten_tomatoes_data(max_words=MAX_WORDS)

    if x_train.size == 0:
        print("Failed to load data or data is empty. Exiting.")
        return

    num_features = x_train.shape[1]
    num_classes = y_train.shape[1]

    print("\n2. Initializing model...")
    model = LogisticRegression(num_features=num_features, num_classes=num_classes)

    print("\n3. Training model...")
    history = model.train(
        x_train, y_train,
        x_dev, y_dev,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    print("\n4. Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 你可以在这里添加代码来绘制 history 或保存模型

if __name__ == '__main__':
    main()
