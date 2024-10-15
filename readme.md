
# AI-Powered Banking Customer Segmentation

This Streamlit application provides an AI-driven customer segmentation and analysis tool for banking institutions. It offers real-time insights into customer behavior, personalized recommendations, and predictive analytics for customer retention.

![Home](https://github.com/user-attachments/assets/0e51fa56-17a3-46d9-8166-77db23356655)

## Features

1. **Customer Profile Overview**:  
   View detailed customer information by entering a customer ID. This feature allows bank personnel to quickly access and analyze specific customer data.

2. **Customer Segmentation Model**:  
   - **Predict customer segments**: The app uses a pre-trained K-means clustering model to assign customers to their respective segments based on their behavioral and demographic data.
   - **Upload CSV data for bulk processing**: Upload CSV files for batch customer segmentation. Ensure the file adheres to the specified format for successful processing.
   - **Input new customer data manually**: Add new customer records by manually entering data through the app.

3. **Real-Time Segmentation Updates**:  
   - **Edit existing customer data**: Modify customer data directly within the app.
   - **Edit customer churn data**: Update customer records for churn analysis.
   - **Delete customer records**: Remove outdated or incorrect records from the database.
   - **Download database**: Export customer segmentation or churn data as CSV files for offline analysis.

4. **Behavioral Analytics for Retention**:  
   - **Predict customer churn probability**: Use the Random Forest model to estimate the likelihood of customer churn based on their transactional and behavioral data.
   - **Generate retention strategies based on customer segments**: Provide retention strategies tailored to customer segments, helping to prevent churn and increase loyalty.

5. **Personalized Banking Offers**:  
   Generate customized banking services and offers for individual customers based on their segments and behaviors, ensuring targeted and relevant recommendations.

## Project Structure

```
AI-Powered-Banking-Customer-Segmentation/
│
├── Records/
│   └── (CSV files of data stored in the database)
│
├── Notebook/
│   ├── (CSV files for analysis)
│   └── (Jupyter notebook files - .ipynb)
│
├── Webapp/
│   ├── app.py
│   ├── .env
│   └── Models/
│       └── (Pre-trained machine learning models)
│
├── requirements.txt
└── README.md
```

- **Records/**: Contains CSV files with data extracted from the database.
- **Notebook/**: Includes CSV files for analysis and Jupyter notebooks used during data exploration and model development.
- **Webapp/**: Contains files for the Streamlit application.
  - **app.py**: The main application script to run the Streamlit interface.
  - **.env**: Stores environment variables for database credentials and configuration.
  - **Models/**: Directory where pre-trained machine learning models for customer segmentation and churn prediction are stored.
- **requirements.txt**: Specifies the required Python packages and dependencies for the project.

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/CodeWizardl/AI-Powered-Banking-Customer-Segmentation.git
```

### Step 2: Navigate to the project directory
```bash
cd AI-Powered-Banking-Customer-Segmentation
```

### Step 3: Create and activate a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate   # on Mac/Linux
venv\Scripts\activate      # on Windows
```

### Step 4: Install the required packages
```bash
pip install -r requirements.txt
```

### Step 5: Set up environment variables
- Navigate to the `Webapp/` directory.
- Create or edit the `.env` file.
- Add the following variables to the `.env` file:
  ```bash
  DB_USER=your_username
  DB_PASSWORD=your_password
  DB_HOST=your_host
  DB_PORT=your_port
  DB_NAME=your_database_name
  ```
- Replace the placeholders with your actual PostgreSQL credentials.

### Step 6: Set up your PostgreSQL database
- Ensure PostgreSQL is installed and running.
- Set up the required tables and schemas as per the application’s needs (e.g., `Customer`, `CustomerChurn`).

### Step 7: Run the Streamlit application
```bash
cd Webapp
streamlit run app.py
```

Open your web browser and go to `http://localhost:8501` to access the application.

## Dependencies

The app relies on the following libraries:

- Streamlit
- Pandas
- SQLAlchemy
- Psycopg2 (for PostgreSQL connection)
- Scikit-learn
- python-dotenv (for loading environment variables)

## Data Models

### Customer
- Stores general customer information, including demographic and transactional details.
- Used for customer segmentation.

### CustomerChurn
- Stores data related to customer churn prediction, allowing churn probability estimations for each customer.

## Machine Learning Models

### Customer Segmentation
- **Algorithm**: K-means clustering
- **Purpose**: Segment customers into distinct groups based on their banking behavior.

### Churn Prediction
- **Algorithms evaluated**:
  - Random Forest
  - Gradient Boosting
  - SVM (Support Vector Machine)
  - KNN (K-Nearest Neighbors)
  - Logistic Regression
  - Decision Tree
- **Chosen Model**: Random Forest, due to its superior performance in predicting churn accurately.
- **Storage**: Pre-trained models are serialized using `pickle` and stored in the `Webapp/Models/` directory.

## Database

- **Backend**: PostgreSQL
- **Interaction**: SQLAlchemy is used for ORM (Object Relational Mapping) to interact with the PostgreSQL database.
- **Setup**: The database connection is managed through environment variables stored in the `.env` file. Ensure your PostgreSQL server is running, and the correct credentials are set before launching the app.

### Database Tables
- **Customer Table**: Stores customer profile information.
- **CustomerChurn Table**: Stores churn prediction data.

## Error Handling

- Common Errors like database connection failures, invalid customer IDs, and CSV format mismatches are handled within the application. If you encounter issues, refer to the logs or console output for troubleshooting.

## Contributing

Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License.
