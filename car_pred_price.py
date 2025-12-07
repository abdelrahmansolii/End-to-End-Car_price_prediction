import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Load model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open(r"C:\Users\abdo_\My_Current_scratching\cars_prediction.sav",'rb'))
        return model
    except:
        st.error("Model file not found. Please ensure 'cars_prediction.sav' is in the same directory.")
        return None

model = load_model()

# Title and description
st.title('🚗 Car Price Prediction')
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
   
    st.sidebar.info('This application predicts car prices based on various features. Fill in all the details on the right and click "Predict Price".')
    
    # Show prediction button in sidebar
    predict_button = st.sidebar.button('🔮 Predict Price', type='primary', use_container_width=True)
    
    # Add some info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("This model was trained on historical car data to predict market prices.")
    st.sidebar.markdown("The prediction is based on features like manufacturer, model, mileage, age, and other specifications.")

with col2:
    # Create two columns for images with better spacing
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.image(
            'https://www.apetogentleman.com/wp-content/uploads/2021/02/classic-cars-2.jpg',
            caption="Classic Cars Collection"
        )
    
    with img_col2:
        st.image(
            'https://spn-sta.spinny.com/blog/20220228142243/ezgif.com-gif-maker-98-5.jpg',
           
            caption="Modern Luxury Cars"
        )
     
# Main content area
st.header('📋 Enter Car Details')

# Create two columns for better layout
col_left, col_right = st.columns(2)

with col_left:
    # Manufacturer
    m1 = ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
           'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN',
           'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA',
           'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT', 'INFINITI',
           'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ',
           'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR',
           'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC',
           'PEUGEOT', 'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL', 'HUMMER', 'SCION',
           'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH',
           'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE',
           'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL']
    m2 = [27, 6, 18, 13, 20, 49, 31, 39, 23, 51, 2, 40, 35, 3, 47, 9, 24,
           34, 46, 30, 15, 12, 21, 48, 36, 0, 8, 33, 11, 26, 50, 22, 45, 7,
           28, 4, 10, 37, 52, 53, 17, 5, 43, 32, 44, 25, 29, 41, 1, 19, 38,
           42, 14, 16]
    Manufacturer_mapping = dict(zip(m1, m2))
    Manufacturer1 = st.selectbox('Manufacturer', m1, index=m1.index('TOYOTA') if 'TOYOTA' in m1 else 0)
    Manufacturer = Manufacturer_mapping[Manufacturer1]
    
    # Model - This needs to be expanded based on your actual data
    # I'm showing a more realistic approach - you should load your actual models
    mm1 = ['RX 450', 'Equinox', 'FIT', 'E 230 124', 'RX 450 F SPORT', 'Prius C aqua', 'Camry', 'Corolla', 'Accord', 'Civic']
    mm2 = [890, 458, 477, 485, 470, 833, 150, 250, 350, 450]
    Model_mapping = dict(zip(mm1, mm2))
    Model1 = st.selectbox('Model', mm1, index=0)
    Model = Model_mapping[Model1]
    
    # Category
    c1 = ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon',
           'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine',
           'Pickup']
    c2 = [4, 3, 9, 2, 10, 7, 0, 1, 6, 8, 5]
    Category_Mapping = dict(zip(c1, c2))
    Category1 = st.selectbox('Category', c1, index=c1.index('Sedan') if 'Sedan' in c1 else 0)
    Category = Category_Mapping[Category1]
    
    # Leather interior
    l1 = ['Yes', 'No']
    l2 = [1, 0]  # Fixed: Yes=1, No=0 (based on your notebook encoding)
    leather_mapping = dict(zip(l1, l2))
    Leather1 = st.selectbox('Leather interior', l1, index=0)
    Leather = leather_mapping[Leather1]
    
    # Fuel type
    f1 = ['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen']
    f2 = [2, 5, 1, 6, 4, 0, 3]
    Fuel_Mapping = dict(zip(f1, f2))
    Fuel1 = st.selectbox('Fuel type', f1, index=f1.index('Petrol') if 'Petrol' in f1 else 0)
    Fuel = Fuel_Mapping[Fuel1]
    
    # Gear box type
    g1 = ['Automatic', 'Tiptronic', 'Variator', 'Manual']
    g2 = [0, 2, 3, 1]
    Gear_mapping = dict(zip(g1, g2))
    Gear1 = st.selectbox('Gear box type', g1, index=0)
    Gear = Gear_mapping[Gear1]

with col_right:
    # Drive wheels
    d1 = ['4x4', 'Front', 'Rear']
    d2 = [0, 1, 2]
    Drive_mapping = dict(zip(d1, d2))
    Drive1 = st.selectbox('Drive wheels', d1, index=1)
    Drive = Drive_mapping[Drive1]
    
    # Wheel
    w1 = ['Left wheel', 'Right-hand drive']
    w2 = [0, 1]
    Wheel_mapping = dict(zip(w1, w2))
    Wheel1 = st.selectbox('Wheel', w1, index=0)
    Wheel = Wheel_mapping[Wheel1]
    
    # Color
    cc1 = ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red',
           'Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige',
           'Carnelian red', 'Purple', 'Pink']
    cc2 = [12, 1, 14, 7, 2, 13, 11, 8, 6, 15, 3, 5, 0, 4, 10, 9]
    color_mapping = dict(zip(cc1, cc2))
    Color1 = st.selectbox('Color', cc1, index=cc1.index('Black') if 'Black' in cc1 else 0)
    Color = color_mapping[Color1]
    
    # Engine volume
    Engine = st.selectbox('Engine volume', 
                          sorted([3.5, 3.0, 1.3, 2.5, 2.0, 1.8, 2.4, 1.6, 2.2, 1.5, 
                                  3.3, 1.4, 2.3, 3.2, 1.2, 1.7, 2.9, 1.9, 2.7, 
                                  2.8, 2.1, 1.0, 0.8, 3.4, 2.6, 1.1]), 
                          index=0)
    
    # Cylinders
    Cylinders = st.selectbox('Cylinders', 
                             sorted([6, 4, 8, 1, 12, 3, 2, 16, 5, 7, 9, 10, 14]), 
                             index=1)  # Default to 4 cylinders
    
    # Airbags
    Airbags = st.selectbox('Airbags', 
                           sorted([12, 8, 2, 0, 4, 6, 10, 3, 1, 16, 5, 7, 9, 11, 14, 15, 13]), 
                           index=5)  # Default to 6 airbags
    
    # Create two columns for number inputs
    num_col1, num_col2 = st.columns(2)
    
    with num_col1:
        # Age
        Age = st.number_input('Car Age (years)', min_value=0, max_value=50, value=5, step=1)
        
        # Mileage
        Mileage = st.number_input('Mileage (km)', min_value=0, max_value=500000, value=50000, step=1000)
    
    with num_col2:
        # Levy
        Levy = st.number_input('Levy', min_value=0, max_value=5000, value=500, step=10)
        
        # Engine Power (from your notebook)
        Engine_power = st.number_input('Engine Power', min_value=0.0, max_value=1000.0, value=100.0, step=5.0)

# Additional features from notebook
st.markdown("---")
st.subheader("📊 Additional Features")

col_extra1, col_extra2 = st.columns(2)

with col_extra1:
    # Price per mileage (calculated field)
    if Mileage > 0:
        Price_per_mileage = st.number_input('Price per Mileage (auto-calculated)', 
                                           value=float(Mileage / 1000), 
                                           disabled=True,
                                           help="Auto-calculated as Mileage/1000. You can adjust if needed.")
    else:
        Price_per_mileage = st.number_input('Price per Mileage', value=0.0)
    
    # Brand mean price (you might want to calculate this or get from database)
    Brand_mean_price = st.number_input('Brand Mean Price', min_value=0, max_value=1000000, value=20000, step=1000)

with col_extra2:
    # Model mean price
    Model_mean_price = st.number_input('Model Mean Price', min_value=0, max_value=1000000, value=25000, step=1000)
    
    # Doors (from your notebook)
    doors_options = ['04-May', '02-Mar', '>5']
    Doors = st.selectbox('Doors', doors_options, index=0)

# Create dataframe with ALL features from your notebook
df = pd.DataFrame({
    'manufacturer': [Manufacturer],
    'model': [Model],
    'category': [Category],
    'leather_interior': [Leather],
    'fuel_type': [Fuel],
    'mileage': [Mileage],
    'gear_box_type': [Gear],
    'drive_wheels': [Drive],  # Added from notebook
    'wheel': [Wheel],
    'color': [Color],  # Added from notebook
    'levy': [Levy],
    'engine_volume': [Engine],
    'cylinders': [Cylinders],
    'airbags': [Airbags],  # Added from notebook
    'car_age': [Age],
    'price_per_mileage': [Price_per_mileage],
    'engine_power': [Engine_power],
    'brand_mean_price': [Brand_mean_price],
    'model_mean_price': [Model_mean_price]
    # Note: 'doors' column is in the notebook but not in the final model features
    # If your model was trained with it, add it: 'doors': [Doors]
}, index=[0])

# Show the input data
with st.expander("📋 View Input Data"):
    st.dataframe(df)

# Prediction section
st.markdown("---")
st.subheader("🎯 Prediction Result")

# Prediction section
st.markdown("---")
st.subheader("🎯 Prediction Result")

if predict_button and model is not None:
    try:
        # Make prediction
        prediction = model.predict(df)
        
        # Format price professionally
        rounded_price = int(round(float(prediction[0])))
        formatted_price = f"${rounded_price:,}"
        
        # Display prediction in a professional way
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 20px 0;
        ">
            <h3 style="margin: 0; font-size: 18px; opacity: 0.9;">PREDICTED CAR PRICE</h3>
            <h1 style="margin: 10px 0; font-size: 48px; font-weight: bold;">{formatted_price}</h1>
            <p style="margin: 0; opacity: 0.8; font-size: 14px;">Based on provided specifications</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add some visualization
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            st.metric("Car Age", f"{Age} years")
        
        with col_pred2:
            st.metric("Mileage", f"{Mileage:,} km")
        
        with col_pred3:
            st.metric("Engine Power", f"{Engine_power} HP")
        
        # Show prediction confidence or additional info
        st.info("💡 **Note**: This prediction is based on historical market data. Actual price may vary based on condition, location, and market trends.")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please check that all input features match what the model was trained on.")
        
elif predict_button and model is None:
    st.error("❌ Model not loaded. Please check the model file.")