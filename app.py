import os
os.system("pip install scikit-learn")


import gradio as gr
import pickle
import numpy as np

# Load trained model
with open('car_price3.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_price(Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner):
    # Convert categorical inputs to numerical
    fuel_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    seller_dict = {'Dealer': 0, 'Individual': 1}
    transmission_dict = {'Manual': 0, 'Automatic': 1}
    
    Fuel_Type = fuel_dict[Fuel_Type]
    Seller_Type = seller_dict[Seller_Type]
    Transmission = transmission_dict[Transmission]
    
    # Prepare input data
    input_data = np.array([[Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner]])
    
    # Predict the price
    predicted_price = model.predict(input_data)[0]
    return f"Estimated Selling Price: {predicted_price:.2f} Lakhs"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Year"),
        gr.Number(label="Present Price (in Lakhs)"),
        gr.Number(label="Kms Driven"),
        gr.Radio(['Petrol', 'Diesel', 'CNG'], label="Fuel Type"),
        gr.Radio(['Dealer', 'Individual'], label="Seller Type"),
        gr.Radio(['Manual', 'Automatic'], label="Transmission"),
        gr.Number(label="Owner (0/1/2/3)")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="Car Price Prediction",
    description="Enter the details of the car to get an estimated selling price",
    allow_flagging="never"  # Disabled flagging
)

# Launch the interface without the share link
interface.launch()
