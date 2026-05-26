from tensorflow.keras.models import load_model
from data_preprocessing import val_data
model = load_model("best_model.keras")
loss,accuracy= model.evaluate(val_data)
print("validation accuracy",accuracy)
print("validation loss ",loss)