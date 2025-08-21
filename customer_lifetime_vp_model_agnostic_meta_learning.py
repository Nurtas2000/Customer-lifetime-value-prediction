import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

# ====================== DATA PREPARATION ======================
data = """customer_id,age,gender,income_tier,first_purchase_date,last_purchase_date,total_purchases,total_spend,avg_purchase_value,purchase_frequency_days,segment,clv_6month
C1001,32,Male,High,2022-01-15,2022-06-10,14,4200.00,300.00,14,Premium,5250.00
C1002,28,Female,Medium,2022-02-20,2022-06-18,9,1350.00,150.00,18,Standard,1800.00
C1003,45,Male,High,2022-01-05,2022-06-22,18,7200.00,400.00,12,Premium,9000.00
C1004,22,Female,Low,2022-03-10,2022-06-15,5,375.00,75.00,25,Budget,500.00
C1005,39,Male,Medium,2022-02-15,2022-06-20,12,2400.00,200.00,15,Standard,3200.00
C1006,51,Female,High,2022-01-10,2022-06-12,16,5600.00,350.00,13,Premium,7000.00
C1007,26,Female,Low,2022-04-05,2022-06-19,3,120.00,40.00,30,Budget,150.00
C1008,33,Male,Medium,2022-03-01,2022-06-21,8,1280.00,160.00,20,Standard,1700.00
C1009,29,Female,High,2022-02-08,2022-06-17,11,4400.00,400.00,16,Premium,5500.00
C1010,42,Male,Low,2022-04-15,2022-06-14,4,200.00,50.00,28,Budget,250.00
C1011,36,Female,Medium,2022-03-15,2022-06-16,7,1050.00,150.00,18,Standard,1400.00
C1012,48,Male,High,2022-01-20,2022-06-22,15,6000.00,400.00,14,Premium,7500.00
C1013,24,Female,Low,2022-05-10,2022-06-18,2,60.00,30.00,35,Budget,75.00
C1014,31,Male,Medium,2022-04-01,2022-06-20,6,900.00,150.00,22,Standard,1200.00
C1015,27,Female,High,2022-03-05,2022-06-15,10,3500.00,350.00,17,Premium,4375.00
C1016,44,Male,Low,2022-05-15,2022-06-12,3,105.00,35.00,30,Budget,130.00
C1017,38,Female,Medium,2022-02-28,2022-06-19,9,1440.00,160.00,19,Standard,1920.00
C1018,50,Male,High,2022-01-25,2022-06-21,13,4550.00,350.00,15,Premium,5687.50
C1019,23,Female,Low,2022-06-01,2022-06-20,1,25.00,25.00,40,Budget,30.00
C1020,35,Male,Medium,2022-03-20,2022-06-22,5,750.00,150.00,24,Standard,1000.00
C1021,40,Female,High,2022-02-10,2022-06-17,12,4800.00,400.00,16,Premium,6000.00
C1022,25,Male,Low,2022-05-20,2022-06-15,2,70.00,35.00,33,Budget,85.00
C1023,30,Female,Medium,2022-04-10,2022-06-18,7,1120.00,160.00,20,Standard,1500.00
C1024,47,Male,High,2022-01-30,2022-06-19,14,4900.00,350.00,14,Premium,6125.00
C1025,21,Female,Low,2022-06-05,2022-06-16,1,30.00,30.00,38,Budget,35.00
C1026,34,Male,Medium,2022-03-25,2022-06-21,8,1280.00,160.00,19,Standard,1700.00
C1027,41,Female,High,2022-02-15,2022-06-22,11,3850.00,350.00,16,Premium,4812.50
C1028,26,Male,Low,2022-05-25,2022-06-14,3,90.00,30.00,28,Budget,110.00
C1029,37,Female,Medium,2022-04-05,2022-06-20,6,960.00,160.00,21,Standard,1280.00
C1030,49,Male,High,2022-01-18,2022-06-17,15,5250.00,350.00,14,Premium,6562.50
C1031,22,Female,Low,2022-06-08,2022-06-19,2,50.00,25.00,32,Budget,60.00
C1032,33,Male,Medium,2022-03-30,2022-06-22,9,1440.00,160.00,18,Standard,1920.00
C1033,28,Female,High,2022-02-22,2022-06-21,10,3500.00,350.00,16,Premium,4375.00
C1034,43,Male,Low,2022-05-30,2022-06-16,4,120.00,30.00,26,Budget,150.00
C1035,31,Female,Medium,2022-04-15,2022-06-18,7,1120.00,160.00,20,Standard,1500.00
C1036,46,Male,High,2022-01-22,2022-06-20,13,4550.00,350.00,15,Premium,5687.50
C1037,24,Female,Low,2022-06-12,2022-06-22,1,20.00,20.00,40,Budget,25.00
C1038,36,Male,Medium,2022-04-20,2022-06-19,8,1280.00,160.00,19,Standard,1700.00
C1039,29,Female,High,2022-03-10,2022-06-17,11,3850.00,350.00,16,Premium,4812.50
C1040,52,Male,Low,2022-05-18,2022-06-14,5,150.00,30.00,24,Budget,190.00
C1041,27,Female,Medium,2022-04-25,2022-06-21,6,960.00,160.00,20,Standard,1280.00
C1042,38,Male,High,2022-02-05,2022-06-22,14,4900.00,350.00,14,Premium,6125.00
C1043,23,Female,Low,2022-06-15,2022-06-20,2,40.00,20.00,35,Budget,50.00
C1044,35,Male,Medium,2022-03-15,2022-06-18,9,1440.00,160.00,18,Standard,1920.00
C1045,40,Female,High,2022-01-28,2022-06-16,12,4200.00,350.00,15,Premium,5250.00
C1046,25,Male,Low,2022-05-22,2022-06-12,3,75.00,25.00,28,Budget,90.00
C1047,32,Female,Medium,2022-04-08,2022-06-19,7,1120.00,160.00,20,Standard,1500.00
C1048,45,Male,High,2022-02-12,2022-06-21,15,5250.00,350.00,14,Premium,6562.50
C1049,20,Female,Low,2022-06-18,2022-06-22,1,15.00,15.00,42,Budget,20.00
C1050,39,Male,Medium,2022-03-05,2022-06-17,10,1600.00,160.00,17,Standard,2133.33"""

# Load data into DataFrame
df = pd.read_csv(pd.compat.StringIO(data))
df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

# Feature Engineering
df['customer_duration'] = (df['last_purchase_date'] - df['first_purchase_date']).dt.days
df['purchase_intensity'] = df['total_purchases'] / df['customer_duration']
df = df.drop(['customer_id', 'first_purchase_date', 'last_purchase_date'], axis=1)

# Preprocessing
numeric_features = ['age', 'total_purchases', 'total_spend', 'avg_purchase_value', 
                   'purchase_frequency_days', 'customer_duration', 'purchase_intensity']
categorical_features = ['gender', 'income_tier', 'segment']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(df.drop('clv_6month', axis=1))
y = df['clv_6month'].values

# Split into meta-train and meta-test (by segments for few-shot learning)
segments = df['segment'].unique()
train_segments, test_segments = train_test_split(segments, test_size=0.2, random_state=42)

X_train = X[df['segment'].isin(train_segments)]
y_train = y[df['segment'].isin(train_segments)]
X_test = X[df['segment'].isin(test_segments)]
y_test = y[df['segment'].isin(test_segments)]

# ====================== MAML IMPLEMENTATION ======================
class MAML:
    def __init__(self, input_shape, alpha=0.001, beta=0.001):
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta    # Outer loop learning rate
        self.model = self.build_model(input_shape)
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)
        
    def build_model(self, input_shape):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        return model
        
    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))  # MSE
        
    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compute_loss(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.meta_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
        
    def adapt(self, support_x, support_y):
        """Perform inner loop adaptation"""
        # Save original weights
        original_weights = [tf.identity(w) for w in self.model.weights]
        
        with tf.GradientTape() as tape:
            y_pred = self.model(support_x, training=True)
            loss = self.compute_loss(support_y, y_pred)
        
        # Compute gradients and update temporary model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        updated_weights = [w - self.alpha * g for w, g in zip(self.model.weights, gradients)]
        
        # Apply temporary updates
        for i in range(len(self.model.weights)):
            self.model.weights[i].assign(updated_weights[i])
            
        return original_weights
        
    def restore_weights(self, original_weights):
        """Restore original weights after adaptation"""
        for i in range(len(self.model.weights)):
            self.model.weights[i].assign(original_weights[i])
            
    def meta_train(self, X_train, y_train, n_episodes=100, n_tasks=5, k_shots=5):
        """Meta-training loop"""
        losses = []
        
        for episode in range(n_episodes):
            episode_loss = 0
            
            for _ in range(n_tasks):
                # Sample a task (customer segment)
                segment = np.random.choice(train_segments)
                segment_mask = (df['segment'] == segment).values & (df.index.isin(range(len(X_train))).values
                segment_indices = np.where(segment_mask)[0]
                
                if len(segment_indices) < 2*k_shots:
                    continue
                    
                # Split into support (adaptation) and query (evaluation) sets
                np.random.shuffle(segment_indices)
                support_idx = segment_indices[:k_shots]
                query_idx = segment_indices[k_shots:2*k_shots]
                
                support_x = X_train[support_idx]
                support_y = y_train[support_idx]
                query_x = X_train[query_idx]
                query_y = y_train[query_idx]
                
                # Inner loop adaptation
                original_weights = self.adapt(support_x, support_y)
                
                # Outer loop optimization
                with tf.GradientTape() as tape:
                    y_pred = self.model(query_x, training=True)
                    loss = self.compute_loss(query_y, y_pred)
                
                # Compute gradients with respect to original weights
                gradients = tape.gradient(loss, original_weights)
                self.meta_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                episode_loss += loss.numpy()
                
                # Restore original weights
                self.restore_weights(original_weights)
            
            avg_loss = episode_loss / n_tasks
            losses.append(avg_loss)
            print(f"Episode {episode + 1}/{n_episodes}, Loss: {avg_loss:.4f}")
            
        return losses
        
    def evaluate(self, X_test, y_test, n_tasks=3, k_shots=5):
        """Evaluate on unseen customer segments"""
        test_losses = []
        
        for _ in range(n_tasks):
            segment = np.random.choice(test_segments)
            segment_mask = (df['segment'] == segment).values & (df.index.isin(range(len(X_test))).values
            segment_indices = np.where(segment_mask)[0]
            
            if len(segment_indices) < 2*k_shots:
                continue
                
            np.random.shuffle(segment_indices)
            support_idx = segment_indices[:k_shots]
            query_idx = segment_indices[k_shots:2*k_shots]
            
            support_x = X_test[support_idx]
            support_y = y_test[support_idx]
            query_x = X_test[query_idx]
            query_y = y_test[query_idx]
            
            # Adapt to new segment
            original_weights = self.adapt(support_x, support_y)
            
            # Evaluate on query set
            y_pred = self.model(query_x, training=False)
            loss = self.compute_loss(query_y, y_pred)
            test_losses.append(loss.numpy())
            
            # Restore weights
            self.restore_weights(original_weights)
            
        return np.mean(test_losses)

# ====================== TRAINING AND EVALUATION ======================
input_shape = X_train.shape[1]
maml = MAML(input_shape)

# Meta-training
print("Starting meta-training...")
train_losses = maml.meta_train(X_train, y_train, n_episodes=100, n_tasks=5, k_shots=5)

# Evaluation on unseen segments
print("\nEvaluating on unseen customer segments...")
test_loss = maml.evaluate(X_test, y_test, n_tasks=3, k_shots=5)
print(f"Average test loss on unseen segments: {test_loss:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title("Meta-Training Loss")
plt.xlabel("Episode")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# Example prediction after adaptation
segment = np.random.choice(test_segments)
segment_indices = np.where((df['segment'] == segment).values & (df.index.isin(range(len(X_test))).values)[0]
support_idx = segment_indices[:5]
query_idx = segment_indices[5:10]

support_x = X_test[support_idx]
support_y = y_test[support_idx]
query_x = X_test[query_idx]
query_y = y_test[query_idx]

# Adapt to new segment
original_weights = maml.adapt(support_x, support_y)

# Make predictions
predictions = maml.model(query_x).numpy().flatten()

print("\nExample predictions after adaptation:")
for true, pred in zip(query_y, predictions):
    print(f"True CLV: {true:.2f}, Predicted CLV: {pred:.2f}")

# Restore weights
maml.restore_weights(original_weights)
