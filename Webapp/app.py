import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, text
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os
import urllib.parse
import io
from sklearn.impute import SimpleImputer
import time

# Load environment variables from .env file
load_dotenv()

# Database setup using environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))  # URL encode the password
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Create the database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Load pre-trained models
scaler = joblib.load('Models/scaler.pkl')
pca_loaded = joblib.load('Models/pca_model.pkl')
kmeans_model = joblib.load('Models/kmeans_model.pkl')

# Load churn model and scaler
churn_model = joblib.load('Models/best_churn_model.pkl')
churn_scaler = joblib.load('Models/scaler_churn.pkl')

# Segment labels
segment_labels = {
    0: "High-Net-Worth Individual",
    1: "Frequent Credit Card User",
    2: "Occasional Spender",
    3: "Low-Activity Account",
    4: "Cash-Heavy User"
}

# Define the Customer data model
class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    customer_id = Column(String, unique=True)
    balance = Column(Float)
    balance_frequency = Column(Float)
    purchases = Column(Float)
    oneoff_purchases = Column(Float)
    installments_purchases = Column(Float)
    cash_advance = Column(Float)
    purchases_frequency = Column(Float)
    oneoff_purchases_frequency = Column(Float)
    purchases_installments_frequency = Column(Float)
    cash_advance_frequency = Column(Float)
    cash_advance_trx = Column(Integer)
    purchases_trx = Column(Integer)
    credit_limit = Column(Float)
    payments = Column(Float)
    minimum_payments = Column(Float)
    prc_full_payment = Column(Float)
    tenure = Column(Integer)
    cluster = Column(Integer)
    segment = Column(String)
    
    # Relationship to CustomerChurn table
    churn_data = relationship("CustomerChurn", back_populates="customer", uselist=False)

# Define the CustomerChurn data model
class CustomerChurn(Base):
    __tablename__ = 'customer_churn'
    id = Column(Integer, primary_key=True)
    customer_id = Column(String, ForeignKey('customers.customer_id'), unique=True)
    surname = Column(String)
    credit_score = Column(Integer)
    geography = Column(String)
    gender = Column(String)
    age = Column(Integer)
    tenure = Column(Integer)
    balance = Column(Float)
    num_of_products = Column(Integer)
    has_cr_card = Column(Integer)
    is_active_member = Column(Integer)
    estimated_salary = Column(Float)
    exited = Column(Integer)  # The churn label
    churn_probability = Column(Float)  # New field for churn probability
    retention_strategy = Column(String)

    # Relationship to the Customer table
    customer = relationship("Customer", back_populates="churn_data")

# Create the table 
Base.metadata.create_all(engine)

# Function to simulate loading content
def load_content():
    st.session_state.loaded = True

# Initialize session state for loading
if 'loaded' not in st.session_state:
    st.session_state.loaded = False

def splash_screen():
    # Create a placeholder for the title
    title_placeholder = st.empty()
    # Display the title
    title_placeholder.title("Banking Customer Segmentation")
    # Wait for 1 seconds
    time.sleep(1)
    # Clear the title
    title_placeholder.empty()

# Splash screen
if not st.session_state.loaded:
    splash_screen()
    load_content()

def preprocess_data(df):
    df = df.apply(lambda x: np.log1p(x))
    return df

def fetch_customer_segments():
    """
    Fetch all customer segment data from the database and return it as a pandas DataFrame.
    """
    # Fetch all customer data from the database
    customers = session.query(Customer).all()

    # If there is customer data, prepare it in a DataFrame format
    if customers:
        data = [{
            "customer_id": customer.customer_id,
            "balance": customer.balance,
            "balance_frequency": customer.balance_frequency,
            "purchases": customer.purchases,
            "oneoff_purchases": customer.oneoff_purchases,
            "installments_purchases": customer.installments_purchases,
            "cash_advance": customer.cash_advance,
            "purchases_frequency": customer.purchases_frequency,
            "oneoff_purchases_frequency": customer.oneoff_purchases_frequency,
            "purchases_installments_frequency": customer.purchases_installments_frequency,
            "cash_advance_frequency": customer.cash_advance_frequency,
            "cash_advance_trx": customer.cash_advance_trx,
            "purchases_trx": customer.purchases_trx,
            "credit_limit": customer.credit_limit,
            "payments": customer.payments,
            "minimum_payments": customer.minimum_payments,
            "prc_full_payment": customer.prc_full_payment,
            "tenure": customer.tenure,
            "cluster": customer.cluster,
            "segment": customer.segment
        } for customer in customers]

        # Convert to pandas DataFrame
        customer_segments_df = pd.DataFrame(data)
        return customer_segments_df

    else:
        # Return an empty DataFrame if no customer data exists
        st.warning("No customer data found.")
        return pd.DataFrame()

def fetch_customer_churn_data():
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT 
                customer_id,
                surname,
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_of_products,
                has_cr_card,
                is_active_member,
                estimated_salary,
                exited,
                churn_probability,
                retention_strategy
            FROM customer_churn
        """))

        churn_data = result.fetchall()
        # Convert to DataFrame
        churn_df = pd.DataFrame(churn_data, columns=[
            "Customer ID", "Surname", "Credit Score", "Geography", "Gender", "Age", 
            "Tenure", "Balance", "Number of Products", "Has Credit Card", 
            "Is Active Member", "Estimated Salary", "Exited", 
            "Churn Probability", "Retention Strategy"
        ])
        
        return churn_df

# Function to store customer data in the database
def store_customer_data(customer_data, customer_id, cluster, segment):
    new_customer = Customer(
        customer_id=str(customer_id),
        balance=float(customer_data['BALANCE'][0]),
        balance_frequency=float(customer_data['BALANCE_FREQUENCY'][0]),
        purchases=float(customer_data['PURCHASES'][0]),
        oneoff_purchases=float(customer_data['ONEOFF_PURCHASES'][0]),
        installments_purchases=float(customer_data['INSTALLMENTS_PURCHASES'][0]),
        cash_advance=float(customer_data['CASH_ADVANCE'][0]),
        purchases_frequency=float(customer_data['PURCHASES_FREQUENCY'][0]),
        oneoff_purchases_frequency=float(customer_data['ONEOFF_PURCHASES_FREQUENCY'][0]),
        purchases_installments_frequency=float(customer_data['PURCHASES_INSTALLMENTS_FREQUENCY'][0]),
        cash_advance_frequency=float(customer_data['CASH_ADVANCE_FREQUENCY'][0]),
        cash_advance_trx=int(customer_data['CASH_ADVANCE_TRX'][0]),
        purchases_trx=int(customer_data['PURCHASES_TRX'][0]),
        credit_limit=float(customer_data['CREDIT_LIMIT'][0]),
        payments=float(customer_data['PAYMENTS'][0]),
        minimum_payments=float(customer_data['MINIMUM_PAYMENTS'][0]),
        prc_full_payment=float(customer_data['PRC_FULL_PAYMENT'][0]),
        tenure=int(customer_data['TENURE'][0]),
        cluster=int(cluster),
        segment=str(segment)
    )
    try:
        session.add(new_customer)
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Error saving customer data: {e}")
    finally:
        session.close()

# Function to check if the customer ID already exists in the database
def is_customer_id_duplicate(customer_id):
    return session.query(Customer).filter_by(customer_id=customer_id).first() is not None

# Function to retrieve customer data based on ID
def get_customer_data(customer_id):
    customer = session.query(Customer).filter_by(customer_id=customer_id).first()
    if customer:
        return customer
    else:
        return None

# Fetch churn-related data for a specific customer
def get_customer_churn_data(customer_id):
    result = session.query(CustomerChurn).filter(CustomerChurn.customer_id == customer_id).first()
    return result

# Define retention strategies based on segment
def get_retention_strategy(segment):
    strategies = {
        0: "Offer premium financial services and exclusive investment options.",
        1: "Provide cashback rewards and discounts on frequent credit card usage.",
        2: "Send personalized offers to encourage higher spending.",
        3: "Engage with loyalty programs to increase customer activity.",
        4: "Provide high-interest savings options to encourage long-term retention."
    }
    return strategies.get(segment, "Offer personalized customer care and financial advice.")

# Function to store data from a DataFrame
def store_data_from_dataframe(df):
    required_columns = [
        'customer_id', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 
        'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
        'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 
        'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
    ]

    if not all(column in df.columns for column in required_columns):
        # Provide download button for the template
        csv_data = pd.DataFrame(columns=required_columns).to_csv(index=False)
        st.error("The uploaded CSV file does not match the required format. Please download the template and follow the structure.")
        st.download_button(
            label="Download CSV Template",
            data=csv_data,
            file_name="customer_data_template.csv",
            mime="text/csv"
        )
        return
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df[required_columns[1:]] = imputer.fit_transform(df[required_columns[1:]])

    for index, row in df.iterrows():
        customer_id = row['customer_id']

        if is_customer_id_duplicate(customer_id):
            st.error(f"Customer ID {customer_id} already exists in the database.")
            continue

        customer_data = {
            "BALANCE": [row['BALANCE']],
            "BALANCE_FREQUENCY": [row['BALANCE_FREQUENCY']],
            "PURCHASES": [row['PURCHASES']],
            "ONEOFF_PURCHASES": [row['ONEOFF_PURCHASES']],
            "INSTALLMENTS_PURCHASES": [row['INSTALLMENTS_PURCHASES']],
            "CASH_ADVANCE": [row['CASH_ADVANCE']],
            "PURCHASES_FREQUENCY": [row['PURCHASES_FREQUENCY']],
            "ONEOFF_PURCHASES_FREQUENCY": [row['ONEOFF_PURCHASES_FREQUENCY']],
            "PURCHASES_INSTALLMENTS_FREQUENCY": [row['PURCHASES_INSTALLMENTS_FREQUENCY']],
            "CASH_ADVANCE_FREQUENCY": [row['CASH_ADVANCE_FREQUENCY']],
            "CASH_ADVANCE_TRX": row['CASH_ADVANCE_TRX'],
            "PURCHASES_TRX": row['PURCHASES_TRX'],
            "CREDIT_LIMIT": [row['CREDIT_LIMIT']],
            "PAYMENTS": [row['PAYMENTS']],
            "MINIMUM_PAYMENTS": [row['MINIMUM_PAYMENTS']],
            "PRC_FULL_PAYMENT": [row['PRC_FULL_PAYMENT']],
            "TENURE": [row['TENURE']]
        }

        df_processed = preprocess_data(pd.DataFrame(customer_data))
        df_scaled = pd.DataFrame(scaler.transform(df_processed), columns=df_processed.columns)
        X_new_pca_4 = pca_loaded.transform(df_scaled)
        clusters = kmeans_model.predict(X_new_pca_4)
        segment = segment_labels[clusters[0]]
        store_customer_data(df_processed, customer_id, clusters[0], segment)
        
def store_customer_churn_data(customer_data):
    # Assuming you're using a SQLAlchemy session
    try:
        with session.begin():
            churn_record = CustomerChurn(
                customer_id=customer_data['customer_id'],
                surname=customer_data['surname'],  # Add surname if needed
                credit_score=customer_data['credit_score'],
                geography=customer_data['geography'],
                gender=customer_data['gender'],
                age=customer_data['age'],
                tenure=customer_data['tenure'],
                balance=customer_data['balance'],
                num_of_products=customer_data['num_of_products'],
                has_cr_card=customer_data['has_cr_card'],
                is_active_member=customer_data['is_active_member'],
                estimated_salary=customer_data['estimated_salary'],
                exited=customer_data['exited'],  # Churn prediction (0 or 1)
                churn_probability=customer_data['churn_probability'],  # Churn probability
                retention_strategy=customer_data['retention_strategy']  # Retention strategy
            )
            session.add(churn_record)
            session.commit()  # Commit the transaction
            print("Customer churn data saved successfully.")
    except Exception as e:
        session.rollback()  # Roll back in case of error
        print(f"Error saving customer churn data: {e}")

def update_customer_data(customer, customer_data):
    """Update the customer object with new data."""
    customer.balance = float(customer_data['BALANCE'][0])
    customer.balance_frequency = float(customer_data['BALANCE_FREQUENCY'][0])
    customer.purchases = float(customer_data['PURCHASES'][0])
    customer.oneoff_purchases = float(customer_data['ONEOFF_PURCHASES'][0])
    customer.installments_purchases = float(customer_data['INSTALLMENTS_PURCHASES'][0])
    customer.cash_advance = float(customer_data['CASH_ADVANCE'][0])
    customer.purchases_frequency = float(customer_data['PURCHASES_FREQUENCY'][0])
    customer.oneoff_purchases_frequency = float(customer_data['ONEOFF_PURCHASES_FREQUENCY'][0])
    customer.purchases_installments_frequency = float(customer_data['PURCHASES_INSTALLMENTS_FREQUENCY'][0])
    customer.cash_advance_frequency = float(customer_data['CASH_ADVANCE_FREQUENCY'][0])
    customer.cash_advance_trx = int(customer_data['CASH_ADVANCE_TRX'])
    customer.purchases_trx = int(customer_data['PURCHASES_TRX'])
    customer.credit_limit = float(customer_data['CREDIT_LIMIT'][0])
    customer.payments = float(customer_data['PAYMENTS'][0])
    customer.minimum_payments = float(customer_data['MINIMUM_PAYMENTS'][0])
    customer.prc_full_payment = float(customer_data['PRC_FULL_PAYMENT'][0])
    customer.tenure = int(customer_data['TENURE'][0])

    df = pd.DataFrame(customer_data)
    df_processed = preprocess_data(df)
    df_scaled = pd.DataFrame(scaler.transform(df_processed), columns=df_processed.columns)
    X_new_pca_4 = pca_loaded.transform(df_scaled)
    clusters = kmeans_model.predict(X_new_pca_4)
    customer.cluster = int(clusters[0])
    customer.segment = segment_labels[clusters[0]]

    try:
        session.commit()
        st.success("Customer data updated successfully.")
    except Exception as e:
        session.rollback()
        st.error(f"Error updating customer data: {e}")
    finally:
        session.close()

def update_customer_churn_data(churn_data, updated_data):
    # Update each field in the churn_data object
    churn_data.surname = updated_data["SURNAME"]
    churn_data.credit_score = updated_data["CREDIT_SCORE"]
    churn_data.geography = updated_data["GEOGRAPHY"]
    churn_data.gender = updated_data["GENDER"]
    churn_data.age = updated_data["AGE"]
    churn_data.tenure = updated_data["TENURE"]
    churn_data.balance = updated_data["BALANCE"]
    churn_data.num_of_products = updated_data["NUM_OF_PRODUCTS"]
    churn_data.has_cr_card = 1 if updated_data["HAS_CR_CARD"] == "Yes" else 0
    churn_data.is_active_member = 1 if updated_data["IS_ACTIVE_MEMBER"] == "Yes" else 0
    churn_data.estimated_salary = updated_data["ESTIMATED_SALARY"]
    
    # Commit the changes to the database
    session.commit()

def download_database(format, df):
    """Generate a downloadable file from the DataFrame based on the specified format."""
    if format == 'csv':
        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        file_data = output.getvalue()
        mime_type = 'text/csv'
        file_name = 'customer_segments.csv'
    elif format == 'excel':
        # Convert DataFrame to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        file_data = output.getvalue()
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_name = 'customer_segments.xlsx'
    else:
        return None, None, None

    return file_data, mime_type, file_name

def download_churn_database(format, df):
    if format == 'csv':
        file_data = df.to_csv(index=False).encode('utf-8')
        mime_type = 'text/csv'
        file_name = 'customer_churn_data.csv'
    else:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Churn Data')
        file_data = output.getvalue()
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_name = 'customer_churn_data.xlsx'

    return file_data, mime_type, file_name


# Define a more detailed personalized offer generation based on customer data
def get_detailed_personalized_offer(customer):
    # Start with basic offers based on segment
    offer_parts = []
    
    if customer.segment == "High-Net-Worth Individual":
        offer_parts.append("Enjoy exclusive wealth management services tailored to grow your investments and maximize your wealth.")
        offer_parts.append("We recommend considering stocks in sectors like technology and renewable energy for high potential returns.")
        offer_parts.append("Additionally, explore our premium investment funds for diversification.")
    elif customer.segment == "Frequent Credit Card User":
        offer_parts.append("Take advantage of enhanced cashback rewards on your everyday spending and access premium credit card features.")
        offer_parts.append("Consider using your rewards for travel or high-value purchases to maximize benefits.")
    elif customer.segment == "Occasional Spender":
        offer_parts.append("Unlock special discounts on high-value purchases and access low-interest loans for larger expenses.")
        offer_parts.append("We suggest a personal loan option with competitive rates to finance any planned large purchases.")
    elif customer.segment == "Low-Activity Account":
        offer_parts.append("Receive incentives for increasing your account activity, including higher interest on your savings.")
        offer_parts.append("Consider setting up automatic transfers to maximize your savings.")
    elif customer.segment == "Cash-Heavy User":
        offer_parts.append("Get personalized advisory services to help you manage your cash effectively and explore lucrative investment opportunities.")
        offer_parts.append("We recommend exploring high-yield savings accounts or short-term bonds for better returns on your cash.")
    else:
        offer_parts.append("Experience personalized financial consultation to discover options that best meet your financial goals.")
        offer_parts.append("Our advisors can assist you in finding the right investment products suited to your needs.")

    # Tailor offers based on credit score
    if customer.credit_score < 600:
        offer_parts.append("Consider applying for a secured credit card to help rebuild your credit score, opening new financial opportunities.")
        offer_parts.append("We also recommend small, manageable loans to gradually build your credit history.")
    elif 600 <= customer.credit_score < 750:
        offer_parts.append("You qualify for a credit limit increase, which can help enhance your purchasing power and financial flexibility.")
        offer_parts.append("Look into personal loans with low-interest rates for any upcoming financial needs.")
    else:
        offer_parts.append("Enjoy exclusive benefits with top-tier credit products, including low interest rates and premium services.")
        offer_parts.append("Consider investing in premium stock options that have shown consistent growth over the years.")

    # Additional conditions based on customer activity and churn probability
    if customer.num_of_products >= 3:
        offer_parts.append("You are eligible for a premium banking bundle that includes fee waivers and preferential rates tailored to your needs.")
        offer_parts.append("Maximize your benefits by leveraging multiple products for greater rewards.")

    if customer.has_cr_card and customer.is_active_member:
        offer_parts.append("As an active credit card holder, you can expand your rewards program to maximize your benefits.")
        offer_parts.append("Consider enrolling in our loyalty program for added perks and discounts.")

    if customer.churn_probability > 0.7:
        offer_parts.append("We value your loyalty! To help you feel more connected with us, we’re offering a personalized retention plan designed just for you, packed with exclusive benefits.")
    else:
        offer_parts.append("You’re doing great! Enjoy ongoing rewards and opportunities designed to enhance your banking experience.")

    # Combine the offer parts into a numbered list
    return "\n".join(f"{i + 1}. {part}" for i, part in enumerate(offer_parts))

def delete_customer_data(customer_id):
    session = Session()
    try:
        # Fetch the customer record and related churn data
        customer = session.query(Customer).filter_by(customer_id=customer_id).first()
        if customer:
            churn_data = customer.churn_data
            if churn_data:
                session.delete(churn_data)  # Delete from CustomerChurn table first
            session.delete(customer)  # Then delete from Customer table
            session.commit()
        else:
            return None
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Main application code
if st.session_state.loaded:
    # Sidebar for AI-Powered Customer Segmentation
    st.sidebar.title("Banking Customer Segmentation")

    # Section Selection
    section = st.sidebar.radio(
        "Select Section",
        ("Customer Profile Overview", "Customer Segmentation Model", "Real-Time Segmentation Updates", "Behavioral Analytics for Retention", "Personalized Banking Offer")
    )
    
    # Customer Profile Overview (Main section)
    if section == "Customer Profile Overview":
        st.header("Customer Profile Overview")
        st.write("Customer can enter a customer ID to view key details such as their segment, balance, and personalized offers. This helps in understanding the customer's profile and making data-driven decisions.")

        # Input field for customer ID
        customer_id_input = st.text_input("Enter Customer ID:")

        # Fetch and display customer details when "Fetch Details" button is clicked
        if st.button("Fetch Details") and customer_id_input:
            with engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        customers.customer_id, 
                        customers.segment, 
                        customers.balance, 
                        customer_churn.credit_score,    -- Ensure this column exists
                        customer_churn.num_of_products, 
                        customer_churn.has_cr_card, 
                        customer_churn.is_active_member, 
                        customer_churn.churn_probability,
                        customer_churn.surname, 
                        customer_churn.gender, 
                        customer_churn.age, 
                        customer_churn.exited, 
                        customer_churn.retention_strategy
                    FROM customers
                    LEFT JOIN customer_churn ON customers.customer_id = customer_churn.customer_id
                    WHERE customers.customer_id = :customer_id
                """), {"customer_id": customer_id_input})

                customer = result.fetchone()


            # Check if customer data is found
            if customer:
                # Display the customer details
                st.markdown(f"<h4>Customer ID: {customer.customer_id}</h4>", unsafe_allow_html=True)
                st.write(f"**Name:** {customer.surname}")
                st.write(f"**Gender:** {customer.gender}")
                st.write(f"**Age:** {customer.age}")
                st.write(f"**Segment:** {customer.segment}")

                # Display customer-friendly churn information
                status = "likely to leave" if customer.exited == 1 else "loyal and staying with us"
                st.write(f"**Status:** The customer is {status}.")
                st.write(f"**Leaving Probability:** {customer.churn_probability:.2%}")

                # Display retention strategy if applicable
                if customer.exited == 1:
                    st.write(f"**Retention Strategy:** {customer.retention_strategy}")
                else:
                    st.write("This customer is currently retained.")

                # Fetch and display personalized recommendation
                st.write(f"**Personalized Recommendation:**")
                personalized_recommendation = get_detailed_personalized_offer(customer)
                st.write(personalized_recommendation)
            
            else:
                st.warning(f"No customer found with ID {customer_id_input}. Please try again.")

    elif section == "Customer Segmentation Model":
        st.sidebar.header("Customer Segmentation Model")
        segmentation_feature = st.sidebar.selectbox(
            "Choose a feature",
            ("Predict Cluster", "Upload CSV Data", "Input New Customer Data")
        )

        # Predict Cluster - aligns with customer segmentation model
        if segmentation_feature == "Predict Cluster":
            st.header("Predict Customer Segment")
            st.write("This feature uses clustering algorithms to predict which segment a customer belongs to based on their data.")
            
            customer_id = st.text_input("Enter Customer ID for Prediction")
            
            if st.button("Predict Cluster"):
                if customer_id:
                    customer = get_customer_data(customer_id)  
                    
                    if customer:
                        # Display the stored segment for the customer
                        st.success(f"Customer ID {customer_id} belongs to the '{customer.segment}' segment.")
                    else:
                        st.warning(f"No data found for Customer ID: {customer_id}")
                else:
                    st.warning("Please enter a valid Customer ID.")

        # Upload CSV Data - allows bulk input of customer data
        elif segmentation_feature == "Upload CSV Data":
            st.header("Upload Customer Data (CSV)")
            st.write("Upload a CSV file to input bulk customer data and update the segmentation model.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if st.button("Submit"):
                    num_rows = len(df)
                    store_data_from_dataframe(df)
                    st.success(f"Successfully added {num_rows} customer records to the segmentation model.")

        # Input New Customer Data
        elif segmentation_feature == "Input New Customer Data":
            st.header("Input New Customer Data")
            st.write("Manually input customer data to add new customers to the segmentation model.")
            customer_id = st.text_input("Customer ID")
            customer_data = {
                "BALANCE": st.number_input("Balance"),
                "BALANCE_FREQUENCY": st.number_input("Balance Frequency"),
                "PURCHASES": st.number_input("Purchases"),
                "ONEOFF_PURCHASES": st.number_input("One-Off Purchases"),
                "INSTALLMENTS_PURCHASES": st.number_input("Installments Purchases"),
                "CASH_ADVANCE": st.number_input("Cash Advance"),
                "PURCHASES_FREQUENCY": st.number_input("Purchases Frequency"),
                "ONEOFF_PURCHASES_FREQUENCY": st.number_input("One-Off Purchases Frequency"),
                "PURCHASES_INSTALLMENTS_FREQUENCY": st.number_input("Purchases Installments Frequency"),
                "CASH_ADVANCE_FREQUENCY": st.number_input("Cash Advance Frequency"),
                "CASH_ADVANCE_TRX": st.number_input("Cash Advance Transactions"),
                "PURCHASES_TRX": st.number_input("Purchases Transactions"),
                "CREDIT_LIMIT": st.number_input("Credit Limit"),
                "PAYMENTS": st.number_input("Payments"),
                "MINIMUM_PAYMENTS": st.number_input("Minimum Payments"),
                "PRC_FULL_PAYMENT": st.number_input("Percent Full Payment"),
                "TENURE": st.number_input("Tenure")
            }

            if st.button("Submit New Customer Data"):
                if is_customer_id_duplicate(customer_id):
                    st.error(f"Customer ID {customer_id} already exists in the database.")
                else:
                    df = pd.DataFrame([customer_data])
                    df_processed = preprocess_data(df)
                    df_scaled = pd.DataFrame(scaler.transform(df_processed), columns=df_processed.columns)
                    X_new_pca_4 = pca_loaded.transform(df_scaled)
                    clusters = kmeans_model.predict(X_new_pca_4)
                    segment = segment_labels[clusters[0]]
                    store_customer_data(df_processed, customer_id, clusters[0], segment)

    elif section == "Real-Time Segmentation Updates":
        st.sidebar.header("Real-Time Segmentation Updates")
        update_feature = st.sidebar.selectbox(
            "Choose an action",
            ("Edit Customer Data", "Edit Customer Churn Data", "Delete Customer Data", "Download Database")
        )

        # Edit Customer Data - enables updating existing customer data
        if update_feature == "Edit Customer Data":
            st.header("Edit Existing Customer Data")
            st.write("Update existing customer data to trigger real-time updates to their assigned segment.")
            customer_id = st.text_input("Customer ID for Edit")
            customer = get_customer_data(customer_id)
            if customer:
                st.write(f"Editing data for customer ID: {customer_id}")
                customer_data = {
                    "BALANCE": st.number_input("Balance", value=customer.balance),
                    "BALANCE_FREQUENCY": st.number_input("Balance Frequency", value=customer.balance_frequency),
                    "PURCHASES": st.number_input("Purchases", value=customer.purchases),
                    "ONEOFF_PURCHASES": st.number_input("One-Off Purchases", value=customer.oneoff_purchases),
                    "INSTALLMENTS_PURCHASES": st.number_input("Installments Purchases", value=customer.installments_purchases),
                    "CASH_ADVANCE": st.number_input("Cash Advance", value=customer.cash_advance),
                    "PURCHASES_FREQUENCY": st.number_input("Purchases Frequency", value=customer.purchases_frequency),
                    "ONEOFF_PURCHASES_FREQUENCY": st.number_input("One-Off Purchases Frequency", value=customer.oneoff_purchases_frequency),
                    "PURCHASES_INSTALLMENTS_FREQUENCY": st.number_input("Purchases Installments Frequency", value=customer.purchases_installments_frequency),
                    "CASH_ADVANCE_FREQUENCY": st.number_input("Cash Advance Frequency", value=customer.cash_advance_frequency),
                    "CASH_ADVANCE_TRX": st.number_input("Cash Advance Transactions", value=customer.cash_advance_trx),
                    "PURCHASES_TRX": st.number_input("Purchases Transactions", value=customer.purchases_trx),
                    "CREDIT_LIMIT": st.number_input("Credit Limit", value=customer.credit_limit),
                    "PAYMENTS": st.number_input("Payments", value=customer.payments),
                    "MINIMUM_PAYMENTS": st.number_input("Minimum Payments", value=customer.minimum_payments),
                    "PRC_FULL_PAYMENT": st.number_input("Percent Full Payment", value=customer.prc_full_payment),
                    "TENURE": st.number_input("Tenure", value=customer.tenure)
                }

                if st.button("Update Customer Data"):
                    update_customer_data(customer, customer_data)
            else:
                if st.button("Check for Customer"):
                    st.warning(f"No data found for customer ID: {customer_id}")
        
        elif update_feature == "Edit Customer Churn Data":
            st.header("Edit Existing Customer Churn Data")
            st.write("Update existing customer churn data for real-time updates.")
            customer_id = st.text_input("Customer ID for Churn Data Edit")
            churn_data = get_customer_churn_data(customer_id)
            
            if churn_data:
                st.write(f"Editing churn data for customer ID: {customer_id}")
                
                churn_data_updated = {
                    "SURNAME": st.text_input("Name", value=churn_data.surname),
                    "CREDIT_SCORE": st.number_input("Credit Score", value=churn_data.credit_score),
                    "GEOGRAPHY": st.text_input("Geography", value=churn_data.geography),
                    "GENDER": st.selectbox("Gender", ("Male", "Female"), index=0 if churn_data.gender == "Male" else 1),
                    "AGE": st.number_input("Age", value=churn_data.age),
                    "TENURE": st.number_input("Tenure", value=churn_data.tenure),
                    "BALANCE": st.number_input("Balance", value=churn_data.balance),
                    "NUM_OF_PRODUCTS": st.number_input("Number of Products", value=churn_data.num_of_products),
                    "HAS_CR_CARD": st.selectbox("Has Credit Card?", ("Yes", "No"), index=1 if churn_data.has_cr_card else 0),
                    "IS_ACTIVE_MEMBER": st.selectbox("Is Active Member?", ("Yes", "No"), index=1 if churn_data.is_active_member else 0),
                    "ESTIMATED_SALARY": st.number_input("Estimated Salary", value=churn_data.estimated_salary),
                }

                if st.button("Update Churn Data"):
                    update_customer_churn_data(churn_data, churn_data_updated)
                    st.success(f"Churn data for customer ID: {customer_id} has been updated.")
            else:
                if st.button("Check for Customer Churn Data"):
                    st.warning(f"No churn data found for customer ID: {customer_id}")


        # Add download section where users can choose which data they want to download
        elif update_feature == "Download Database":
            st.header("Download Customer Data")
            st.write("Select the type of data you want to download (Customer Segments or Customer Churn).")

            # Let the user choose which type of data to download
            data_type = st.selectbox("Select Data Type", ("Customer Segments", "Customer Churn"))

            if data_type == "Customer Segments":
                # Fetch the customer segments data
                customer_segments_df = fetch_customer_segments()

                if not customer_segments_df.empty:
                    # Show the customer segments data in tabular format
                    st.dataframe(customer_segments_df)

                    # Select format for download
                    download_format = st.selectbox("Select Format", ("CSV", "Excel"))

                    if st.button("Download Segments Data"):
                        format = 'csv' if download_format == "CSV" else 'excel'
                        file_data, mime_type, file_name = download_database(format, customer_segments_df)
                        if file_data:
                            st.download_button(
                                label=f"Download {download_format}",
                                data=file_data,
                                file_name=file_name,
                                mime=mime_type
                            )
                else:
                    st.warning("No data available in the database for Customer Segments.")  # pragma: no cover

            elif data_type == "Customer Churn":
                # Fetch the customer churn data
                customer_churn_df = fetch_customer_churn_data()

                if not customer_churn_df.empty:
                    # Show the customer churn data in tabular format
                    st.dataframe(customer_churn_df)

                    # Select format for download
                    download_format = st.selectbox("Select Format", ("CSV", "Excel"))

                    if st.button("Download Churn Data"):
                        format = 'csv' if download_format == "CSV" else 'excel'
                        file_data, mime_type, file_name = download_churn_database(format, customer_churn_df)
                        if file_data:
                            st.download_button(
                                label=f"Download {download_format}",
                                data=file_data,
                                file_name=file_name,
                                mime=mime_type
                            )
                else:
                    st.warning("No data available in the database for Customer Churn.")
        
        # Add the Delete Customer Data feature
        elif update_feature == "Delete Customer Data":
            st.header("Delete Customer Data")
            customer_id = st.text_input("Customer ID for Deletion")
            customer = get_customer_data(customer_id)

            if customer:
                st.write(f"Customer ID {customer_id} found.")
                if st.button("Delete Customer Data"):
                    delete_customer_data(customer_id)
                    st.success(f"Customer ID {customer_id} has been deleted from the database.")
            else:
                if st.button("Check for Customer"):
                    st.warning(f"No data found for customer ID: {customer_id}")

    
    # Predict Churn with Retention Strategies - New Feature
    elif section == "Behavioral Analytics for Retention":
        st.sidebar.header("Behavioral Analytics for Retention")
        input_mode = st.sidebar.selectbox("Input Mode", ("Enter Customer ID", "Manual Input"))

        if input_mode == "Enter Customer ID":
            st.header("Behavioral Analytics for Retention")
            customer_id = st.text_input("Enter Customer ID for Churn Prediction with Retention Strategies")

            if st.button("Analyze Retention"):
                if customer_id:
                    customer_churn = get_customer_churn_data(customer_id)  # Fetch churn-related data
                    
                    if customer_churn:
                        # Prepare data for churn prediction
                        input_dict = {
                            'CreditScore': customer_churn.credit_score,
                            'Geography': customer_churn.geography,
                            'Gender': customer_churn.gender,
                            'Age': customer_churn.age,
                            'Tenure': customer_churn.tenure,
                            'Balance': customer_churn.balance,
                            'NumOfProducts': customer_churn.num_of_products,
                            'HasCrCard': customer_churn.has_cr_card,
                            'IsActiveMember': customer_churn.is_active_member,
                            'EstimatedSalary': customer_churn.estimated_salary
                        }

                        input_data = pd.DataFrame([input_dict])

                        # One-hot encode categorical features
                        input_data['Geography_Chennai'] = (input_data['Geography'] == 'Chennai').astype(int)
                        input_data['Geography_Hyderabad'] = (input_data['Geography'] == 'Hyderabad').astype(int)
                        input_data = input_data.drop(columns=['Geography'])
                        input_data['Gender_Male'] = (input_data['Gender'] == 'Male').astype(int)
                        input_data = input_data.drop(columns=['Gender'])

                        # Align the input features with the model's expected features
                        expected_features = churn_scaler.feature_names_in_
                        input_data = input_data.reindex(columns=expected_features)

                        # Scale the input data
                        input_data_scaled = churn_scaler.transform(input_data)

                        # Make churn prediction
                        churn_prediction = churn_model.predict(input_data_scaled)
                        churn_proba = churn_model.predict_proba(input_data_scaled)
                        churn_proba = churn_proba[0][1]

                        if churn_prediction[0] == 1:
                            st.write(f"Customer ID {customer_id} is **likely to churn**.")

                            # Provide retention strategy based on segment
                            retention_strategy = get_retention_strategy(Customer.cluster)
                            st.write(f"**Recommended Retention Strategy:** {retention_strategy}")

                        else:
                            st.write(f"Customer ID {customer_id} is **unlikely to churn**.")
                            retention_strategy = None
                        
                        # Store the results in the 
                        churn_prediction = int(churn_prediction[0])
                        customer_churn.churn_probability = churn_proba
                        customer_churn.retention_strategy = retention_strategy
                        customer_churn.exited = churn_prediction  # Store churn status as 0 or 1
                        
                        try:
                            session.commit()  # Save changes to the database
                        except Exception as e:
                            session.rollback()
                            
                    else:
                        st.warning(f"No churn data found for Customer ID: {customer_id}")
                else:
                    st.warning("Please enter a valid Customer ID.")
        
        elif input_mode == "Manual Input":
            # Manual Input Section
            st.write("### Enter Customer Data Manually")

            # Input fields for manual customer details
            customer_data = {
                'customer_id': st.text_input("Customer ID"),
                'surname': st.text_input("Name"),
                'credit_score': st.number_input("Credit Score", min_value=300, max_value=900),
                'geography': st.selectbox("Geography", ["Chennai", "Hyderabad", "Other"]),
                'gender': st.selectbox("Gender", ["Male", "Female"]),
                'age': st.number_input("Age", min_value=18, max_value=100),
                'tenure': st.number_input("Tenure (years)", min_value=0, max_value=10),
                'balance': st.number_input("Balance", min_value=0.0, step=1000.0),
                'num_of_products': st.number_input("Number of Products", min_value=1, max_value=4),
                'has_cr_card': st.selectbox("Has Credit Card", [0, 1]),
                'is_active_member': st.selectbox("Is Active Member", [0, 1]),
                'estimated_salary': st.number_input("Estimated Salary", min_value=0.0, step=1000.0),
                'exited': None,  # Will be predicted later
                'churn_probability': None, 
                'retention_strategy': None
            }

            if st.button("Save Customer Data"):
                if customer_data['customer_id']:
                    # Save the input data to the database
                    store_customer_churn_data(customer_data)
                    st.success("Customer data has been successfully saved to the database.")
                else:
                    st.warning("Please enter a valid Customer ID.")
    
    # If the user selects "Personalized Banking Offer"
    elif section == "Personalized Banking Offer":
        st.header("Personalized Banking Offer")
        st.write("Discover tailored banking services and offers designed to meet your individual financial needs.")

         # Input field for customer ID
        customer_id_input = st.text_input("Enter Customer ID:")
        submit_button = st.button("Fetch Offer")

        if submit_button and customer_id_input:
            # Fetch the data from the database for the entered customer ID
            with engine.connect() as connection:
                # Join Customer and CustomerChurn data to retrieve relevant info using table references
                result = connection.execute(text(f"""
                    SELECT 
                        customers.customer_id, 
                        customers.segment, 
                        customers.balance, 
                        customer_churn.credit_score, 
                        customer_churn.num_of_products, 
                        customer_churn.has_cr_card, 
                        customer_churn.is_active_member, 
                        customer_churn.churn_probability
                    FROM customers
                    LEFT JOIN customer_churn ON customers.customer_id = customer_churn.customer_id
                    WHERE customers.customer_id = :customer_id
                """), {"customer_id": customer_id_input})

                customer = result.fetchone()  # Fetch a single record

            # Check if customer data is returned
            if customer:

                # Display the personalized offer as plain text
                offer = get_detailed_personalized_offer(customer)
                st.write(offer)  
            else:    
                st.write("No customer found with the given ID.")