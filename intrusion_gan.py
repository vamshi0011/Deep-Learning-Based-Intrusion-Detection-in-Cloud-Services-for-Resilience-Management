import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, optimizers

# Dataset path
data_path = "C:/Users/Vamshi/Desktop/Intrusion_detection/data/"

splits = [
    (f"{data_path}train_80.csv", f"{data_path}test_20.csv"),
    (f"{data_path}train_70.csv", f"{data_path}test_30.csv"),
    (f"{data_path}train_50.csv", f"{data_path}test_50.csv"),
    (f"{data_path}train_60.csv", f"{data_path}test_40.csv")
]

# Function to check if files exist
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: Missing file {file_path}")
        return False
    return True

# Function to preprocess dataset
def preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.columns = [col.strip("'") for col in train.columns]
    test.columns = [col.strip("'") for col in test.columns]

    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Ensure consistent encoding for training & testing data
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined.astype(str))
        
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Define GAN components
def build_generator(input_dim):
    model = models.Sequential([
        layers.Dense(16, activation='relu', input_dim=input_dim),
        layers.Dense(input_dim, activation='tanh')
    ])
    return model

def build_discriminator(input_dim):
    model = models.Sequential([
        layers.Dense(16, activation='relu', input_dim=input_dim),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

# Train GAN
def train_gan(generator, discriminator, X_train, epochs=10, batch_size=32):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_data = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, generator.input_shape[1]))
        fake_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((half_batch, 1)))

        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        g_loss = discriminator.train_on_batch(generator.predict(noise), np.ones((batch_size, 1)))

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: D Loss = {0.5 * (d_loss_real[0] + d_loss_fake[0])}, G Loss = {g_loss}")

    return generator.predict(np.random.normal(0, 1, (X_train.shape[0], generator.input_shape[1])))

# Execute for all dataset splits
results = []
for train_file, test_file in splits:
    if not check_file_exists(train_file) or not check_file_exists(test_file):
        continue  # Skip missing datasets

    X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file)

    generator = build_generator(X_train.shape[1])
    discriminator = build_discriminator(X_train.shape[1])
    synthetic_data = train_gan(generator, discriminator, X_train)

    X_combined = np.vstack([X_train, synthetic_data])
    y_combined = np.hstack([y_train, y_train])

    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_combined, y_combined)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy for {train_file}: {accuracy}")
    results.append({'split': train_file, 'accuracy': accuracy})

pd.DataFrame(results).to_csv(f"{data_path}gan_results_comparison.csv", index=False)
print("\n✅ GAN Results saved to", f"{data_path}gan_results_comparison.csv")
