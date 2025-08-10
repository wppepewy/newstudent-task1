import numpy as np

class LogisticRegression:
    def __init__(self, num_features, num_classes):
        """
        初始化模型参数
        :param num_features: 特征数量 (词汇表大小)
        :param num_classes: 类别数量
        """
        self.W = np.zeros((num_features, num_classes))
        self.b = np.zeros(num_classes)

    def softmax(self, z):
        """
        Softmax函数
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        """
        计算交叉熵损失
        :param y_true: 真实标签 (one-hot)
        :param y_pred: 预测概率
        """
        # 防止log(0)
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def train(self, x_train, y_train, x_dev, y_dev, learning_rate=0.1, epochs=10, batch_size=64):
        """
        训练模型
        """
        num_samples = x_train.shape[0]
        history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

        for epoch in range(epochs):
            # Shuffle the training data
            shuffle_indices = np.random.permutation(num_samples)
            x_shuffled = x_train[shuffle_indices]
            y_shuffled = y_train[shuffle_indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                scores = np.dot(x_batch, self.W) + self.b
                y_pred = self.softmax(scores)
                
                # 计算梯度
                grad_W = np.dot(x_batch.T, (y_pred - y_batch)) / batch_size
                grad_b = np.mean(y_pred - y_batch, axis=0)
                
                # 更新参数
                self.W -= learning_rate * grad_W
                self.b -= learning_rate * grad_b

            # 每个epoch结束后评估模型
            train_loss, train_acc = self.evaluate(x_train, y_train)
            dev_loss, dev_acc = self.evaluate(x_dev, y_dev)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['dev_loss'].append(dev_loss)
            history['dev_acc'].append(dev_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")
        
        return history

    def predict(self, x):
        """
        进行预测
        """
        scores = np.dot(x, self.W) + self.b
        return self.softmax(scores)

    def evaluate(self, x, y_true):
        """
        评估模型性能
        """
        y_pred_probs = self.predict(x)
        loss = self.compute_loss(y_true, y_pred_probs)
        
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        
        accuracy = np.mean(y_pred_labels == y_true_labels)
        
        return loss, accuracy

if __name__ == '__main__':
    # 简单的测试
    num_features = 100
    num_classes = 2
    num_samples = 500

    # 生成一些假数据
    X_train_dummy = np.random.rand(num_samples, num_features)
    Y_train_dummy = np.zeros((num_samples, num_classes))
    Y_train_dummy[np.arange(num_samples), np.random.randint(0, num_classes, num_samples)] = 1
    
    X_dev_dummy = np.random.rand(100, num_features)
    Y_dev_dummy = np.zeros((100, num_classes))
    Y_dev_dummy[np.arange(100), np.random.randint(0, num_classes, 100)] = 1

    # 初始化和训练模型
    model = LogisticRegression(num_features=num_features, num_classes=num_classes)
    model.train(X_train_dummy, Y_train_dummy, X_dev_dummy, Y_dev_dummy, epochs=5)
    
    # 评估
    loss, acc = model.evaluate(X_dev_dummy, Y_dev_dummy)
    print(f"\nFinal Dev Accuracy: {acc:.4f}")
