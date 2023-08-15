class AdaptiveHuberLossWithReg(tf.keras.losses.Loss):
    def __init__(self, initial_delta=1.0, lambda_reg=0.01, reduction=tf.keras.losses.Reduction.AUTO, name='adaptive_huber_with_reg'):
        super().__init__(reduction=reduction, name=name)
        self.delta = tf.Variable(initial_value=initial_delta, trainable=False)  # Set trainable to False
        self.lambda_reg = lambda_reg  # Regularization strength
        self.delta_optimizer = tf.keras.optimizers.Adam()  # Use Adam optimizer to update delta

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.delta
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.delta * (tf.abs(error) - 0.5 * self.delta)
        loss = tf.where(is_small_error, small_error_loss, big_error_loss)

        # Add L1 regularization
        loss += self.lambda_reg * tf.abs(self.delta)

        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_true = data[1]
            y_pred = self.call(data[0])
            loss = self(y_true, y_pred)  # Compute the loss value

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update delta using its own optimizer
        delta_gradient = gradients[-1]  # Assume the gradient for delta is the last one
        self.delta_optimizer.apply_gradients(zip([delta_gradient], [self.delta]))

        # Update the other weights using the model's optimizer
        self.optimizer.apply_gradients(zip(gradients[:-1], self.trainable_variables[:-1]))

        return {"loss": loss}

tf.keras.utils.get_custom_objects()['adaptive_huber_with_reg'] = AdaptiveHuberLossWithReg
