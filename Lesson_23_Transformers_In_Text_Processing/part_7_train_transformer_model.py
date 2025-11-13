def train_transformer_model(model, dataset, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        for (inputs, targets) in dataset:
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')
