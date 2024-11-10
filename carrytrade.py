import streamlit as st
import numpy as np
import pandas as pd
import datetime
from datetime import date
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt



# Custom CSS to enhance page appearance
st.markdown("""
    <style>
    .main-container {
        max-width: 1200px;
        margin: auto;
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .header-text {
        font-size: 2em;
        font-weight: bold;
        color: #004080;
        text-align: center;
        margin-bottom: 20px;
    }
    .formula-text {
        font-size: 1.2em;
        text-align: justify;
        margin-top: 30px;
    }
    .button-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #004080;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Go to Paper button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button('Go to Paper'):
    st.markdown('<meta http-equiv="refresh" content="0;url=https://example.com">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)


# Main app functionality (existing logic)




generate_colors = lambda n: [plt.cm.tab10(i / n) for i in range(n)]


df = pd.read_excel("./fred_codes.xlsx",sheet_name="FRED codes")
portfolio = pd.read_excel("./fred_codes.xlsx",sheet_name="Portfolio")

DFS = []
countries = df['Country'].unique()
asset_classes = ["Stock Index", "Bond", "REIT"]  # Available asset classes

df = pd.read_excel("./fred_codes.xlsx",sheet_name="FRED codes")
days90 = df[df['Name'].str.contains("3-Month or 90-Day ", case=False, na=False)]

stock = portfolio[["Country","Stock Index"]][portfolio['Stock Index']!=999]
bond = portfolio[["Country","Bond"]][portfolio['Bond']!=999]
reit = portfolio[["Country","REIT"]][portfolio['REIT']!=999]





# Initialize session state variables
if 'investment_asset_classes' not in st.session_state:
    st.session_state.investment_asset_classes = []  # List to store selected countries and asset classes
if 'borrowing_country' not in st.session_state:
    st.session_state.borrowing_country = None
if 'investment_countries' not in st.session_state:
    st.session_state.investment_countries = []
if 'borrowing_countries' not in st.session_state:
    st.session_state.borrowing_countries = []
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False
if 'weights' not in st.session_state:
    st.session_state.weights = []  # Store weights for each asset class
if 'start_date' not in st.session_state:
    st.session_state.start_date = None
if 'end_date' not in st.session_state:
    st.session_state.end_date = None

# Function to reset session state
def reset_session():
    st.session_state.investment_asset_classes = []
    st.session_state.borrowing_country = None
    st.session_state.investment_countries = []
    st.session_state.borrowing_countries = []
    st.session_state.confirmed = False
    st.session_state.weights = []
    st.session_state.start_date = None
    st.session_state.end_date = None
    st.experimental_rerun()  # Force rerun to clear the state in the UI

# Function to dynamically create country pairs (first tab)
def create_pairs():



    st.markdown("<h1 style='text-align: center;'>Investment and Borrowing Pair Selector</h1>", unsafe_allow_html=True)

    st.write("### Select countries to invest in and borrow from:")

    # Form to add new pairs, with all countries reselectable
    if not st.session_state.confirmed:
        with st.form(key="pair_form"):
            invest_country = st.selectbox("Choose a country to invest in:", countries, key='invest')
            borrow_country = st.selectbox("Choose a country to borrow from:", countries, key='borrow')
            submit = st.form_submit_button("Add Pair")

            if submit:
                # Append only if selections are valid and not equal
                if invest_country and borrow_country and invest_country != borrow_country:
                    st.session_state.investment_countries.append(invest_country)
                    st.session_state.borrowing_countries.append(borrow_country)

                # Refresh selections for subsequent pairs
                st.experimental_rerun()

    # Display the final table with 'Invest' and 'Borrow' columns
    if len(st.session_state.investment_countries) > 0:
        st.markdown("<h3 style='text-align: center;'>Final Investment and Borrowing Table</h3>", unsafe_allow_html=True)
        
        # Create a dataframe for display
        pairs_df = pd.DataFrame({
            "Invest": st.session_state.investment_countries,
            "Borrow": st.session_state.borrowing_countries
        })

        # Center the table with inline styling
        st.markdown(
            f'<div style="display: flex; justify-content: center;">{pairs_df.to_html(index=False)}</div>',
            unsafe_allow_html=True
        )

        # Option to remove a pair
        if not st.session_state.confirmed:
            st.write("### Remove a Pair:")
            pair_to_remove = st.selectbox("Select pair to remove", 
                                          [f"Invest: {i}, Borrow: {b}" for i, b in zip(st.session_state.investment_countries, st.session_state.borrowing_countries)], key='remove')

            if st.button("Remove Selected Pair"):
                # Extract the indices of the selected pair
                idx_to_remove = [f"Invest: {i}, Borrow: {b}" for i, b in zip(st.session_state.investment_countries, st.session_state.borrowing_countries)].index(pair_to_remove)

                # Remove the selected pair from session state lists
                st.session_state.investment_countries.pop(idx_to_remove)
                st.session_state.borrowing_countries.pop(idx_to_remove)
                st.experimental_rerun()  # Refresh the app to update the table and selections

        # Add a date input before confirmation with valid default dates
        st.write("### Enter the start and end date for the investment period:")
        today = datetime.date.today()
        date_range = st.date_input("Select a date range", value=(today-datetime.timedelta(365*5), today), key='date_range_1')

        if date_range and len(date_range) == 2:
            st.session_state.start_date, st.session_state.end_date = date_range
        
        # Add a confirm button to finalize choices only if dates are selected
        if st.session_state.start_date and st.session_state.end_date:
            if st.button("Confirm Selections"):
                st.session_state.confirmed = True
                st.experimental_rerun()  # Re-render the page in "confirmed" state

    # Once confirmed, display the final table and print it to the console
    if st.session_state.confirmed:
        st.markdown("<h3 style='text-align: center; color: green;'>Your Selections Have Been Confirmed!</h3>", unsafe_allow_html=True)


        # Capture selections in a dictionary
        confirmed_pairs = {
            "Investment Countries": st.session_state.investment_countries,
            "Borrowing Countries": st.session_state.borrowing_countries,
            "Start Date": st.session_state.start_date,
            "End Date": st.session_state.end_date
        }



        df = pd.read_excel("./fred_codes.xlsx",sheet_name="FRED codes")
        days90 = df[df['Name'].str.contains("3-Month or 90-Day ", case=False, na=False)]
        overnight = df[df['Name'].str.contains("Call ", case=False, na=False)]


        choice = pd.DataFrame(confirmed_pairs)

        choice = choice.merge(days90[['Country', 'Currency']], left_on='Investment Countries', right_on='Country', how='left').rename(columns={'Currency': 'invest_currency'})

        choice = choice.merge(days90[['Country', 'Currency']], left_on='Borrowing Countries', right_on='Country', how='left').rename(columns={'Currency': 'borrow_currency'})

        choice['exchange_pair'] = choice['invest_currency'] + choice['borrow_currency']


        my_start_date = choice['Start Date'].iloc[0]
        my_end_date = choice['End Date'].iloc[0]


        choice = choice.drop(columns=['Start Date', 'End Date',
                                     'invest_currency','borrow_currency',
                                     'Country_x','Country_y'])



        choice = choice.merge(days90[['Country', 'Code']], left_on='Investment Countries', right_on='Country', how='left').rename(columns={'Code': 'Investment Code'})
        choice = choice.merge(days90[['Country', 'Code']], left_on='Borrowing Countries', right_on='Country', how='left').rename(columns={'Code': 'Borrowing Code'})
        choice = choice.drop(columns=['Country_x', 'Country_y'])
        choice['Label'] =  choice['Investment Countries'] + " vs. " + choice['Borrowing Countries']



        #if (my_start_date - my_end_date == datetime.timedelta(0)) or (my_end_date>date.today()):
        if (my_start_date - my_end_date == datetime.timedelta(0)) or (my_end_date > date.today()):            
            my_end_date = date.today()
            my_start_date = my_end_date - datetime.timedelta(365)


        col_sets = []
        for index, row in choice.iterrows():
            if not yf.download(f"{row['exchange_pair']}=X", start=my_start_date.strftime('%Y-%m-%d'), end=my_end_date.strftime('%Y-%m-%d'), interval='1d')['Close'].empty:
                col_sets.append([row['Investment Code'],row['Borrowing Code'],row['exchange_pair'],row['Label']])

     

        for i, (invest_col, borrow_col, ticker, label) in enumerate(col_sets):
            # Amount Invested is fixed at 1
            amount_invested = 1





            fred_api_key = st.secrets["FRED_API_KEY"]
            fred = Fred(api_key=fred_api_key)

            series_ids = [invest_col, borrow_col]
            data = {}
            for series_id in series_ids:
                data[series_id] = fred.get_series(series_id)
            df = pd.DataFrame(data)
            df['Date'] = df.index
            df['Date'] = df['Date'].dt.date


            if (df[(df['Date'] > my_start_date) & (df['Date'] < my_end_date)].empty) or (my_end_date>date.today()):
                my_end_date = date.today()
                my_start_date = my_end_date - datetime.timedelta(365)
                
                

            df = df[(df['Date'] > my_start_date) & (df['Date'] < my_end_date)].dropna()


            tickerx = f"{ticker}=X"
            data = yf.download(tickerx, start=my_start_date.strftime('%Y-%m-%d'), end=my_end_date.strftime('%Y-%m-%d'), interval='1d')['Close']


            data = data.reset_index()

            data['Date'] = pd.to_datetime(data['Date']).dt.date


            merged_df = pd.merge(data, df, on='Date', how='left')
            merged_df = merged_df.ffill()

            merged_df = merged_df.rename(columns={tickerx: ticker,})
            merged = merged_df.dropna()

            
            merged['S_t+T'] = merged[ticker].shift(1)

            # Drop the last row because S_{t+T} will be NaN for it
            merged_filtered = merged.dropna()
            T=90
            # Calculate the profit for each row
            merged_filtered['Profit %'] = 1 * (
                ( #1+
                    (merged_filtered[invest_col] - merged_filtered[borrow_col])/100 *
                    (merged_filtered[ticker] / merged_filtered['S_t+T'])
                )#** (T / 90) - 1
            )


            DFS.append(merged_filtered[['Date', 'Profit %']].rename(columns={'Profit %': label}))


        if len(DFS) == 1:
            merged_filtered.iloc[:, 1:] = merged_filtered.iloc[:, 1:].applymap(lambda x: x * 100)  # Converting values to percentages
            st.line_chart(
                merged_filtered,
                x="Date",
                y="Profit %")

        else:


            dfs = pd.concat(DFS, axis=1, join='inner')
            dfs = dfs.loc[:, ~dfs.columns.duplicated()]
            
            dfs.iloc[:, 1:] = dfs.iloc[:, 1:].applymap(lambda x: x * 100)  # Converting values to percentages

            st.line_chart(
                dfs,
                x="Date",
                y=list(dfs.columns[1:]),
                color=generate_colors(len(dfs.columns[1:])),
                width=1200,  # Adjust height if needed
                height=800,  # Adjust height if needed
                use_container_width=True  # Use full width of the container
            )





        #dfs.iloc[:, 1:] = dfs.iloc[:, 1:].applymap(lambda x: x * 100)  # Converting values to percentages"""
 

   














# Function to create country-asset selection with weights (second tab)
def create_investment_asset_selection():
    st.markdown("<h1 style='text-align: center;'>Country to Borrow From and Asset Class Selector with Weights</h1>", unsafe_allow_html=True)

    # Form to add new country-asset-weight combinations
    if not st.session_state.confirmed:
        with st.form(key="what"):
            st.write("### Select a country to borrow from and up to 3 asset classes with their weights:")

            # Select a country to borrow from (moved to this tab)
            borrowing_country = st.selectbox("Choose a country to borrow from:", countries, key='borrow_from')
            if borrowing_country:
                st.session_state.borrowing_country = borrowing_country



            df = pd.read_excel("./fred_codes.xlsx",sheet_name="FRED codes")
            days90 = df[df['Name'].str.contains("3-Month or 90-Day ", case=False, na=False)]
            portfolio = pd.read_excel("./fred_codes.xlsx",sheet_name="Portfolio")
            stock = portfolio[["Country","Stock Index"]][portfolio['Stock Index']!=999]
            bond = portfolio[["Country","Bond"]][portfolio['Bond']!=999]
            reit = portfolio[["Country","REIT"]][portfolio['REIT']!=999]


            asset_class_1 = st.selectbox("Select the country for Stock Index:", list(stock['Country']), key='asset_class_1')
            weight_1 = st.number_input("Enter weight for the Stock Index (out of 100):", min_value=0, max_value=100, key='weight_1')

            asset_class_2 = st.selectbox("Select the country for Bond:", list(bond['Country']), key='asset_class_2')
            weight_2 = st.number_input("Enter weight for the Bond (out of 100):", min_value=0, max_value=100, key='weight_2')

            asset_class_3 = st.selectbox("Select the country for REIT:", list(reit['Country']), key='asset_class_3')
            weight_3 = st.number_input("Enter weight for the REIT (out of 100):", min_value=0, max_value=100, key='weight_3')



            submit_assets = st.form_submit_button("Confirm Choice")

            if submit_assets:
                total_weight = weight_1 + weight_2 + weight_3

                # Ensure total weight is exactly 100%
                if total_weight == 100:
                    st.session_state.investment_asset_classes = [
                        {"country": asset_class_1, "asset_class": "Stock Index", "weight": weight_1},
                        {"country": asset_class_2, "asset_class": "Bond", "weight": weight_2},
                        {"country": asset_class_3, "asset_class": "REIT", "weight": weight_3}
                    ]



                else:
                    st.warning("Total weights must sum to 100%. Please adjust your inputs.")

    # Display the selected country-asset-weight combinations
    if st.session_state.investment_asset_classes:
        st.markdown("<h3 style='text-align: center;'>Selected Borrowing Country and Investment Asset Classes with Weights</h3>", unsafe_allow_html=True)

        # Display the borrowing country
        st.write(f"**Borrowing Country:** {st.session_state.borrowing_country}")

        # Create a dataframe to display the selected assets and weights
        assets_df = pd.DataFrame(st.session_state.investment_asset_classes)

        # Center the table with inline styling
        st.markdown(
            f'<div style="display: flex; justify-content: center;">{assets_df.to_html(index=False)}</div>',
            unsafe_allow_html=True
        )

    # Add a date input before confirmation with valid default dates
    st.write("### Enter the start and end date for the investment period:")
    today = datetime.date.today()
    date_range = st.date_input("Select a date range", value=(today-datetime.timedelta(365*5), today), key='date_range_2')  

    if date_range and len(date_range) == 2:
        st.session_state.start_date, st.session_state.end_date = date_range

    # Add a confirm button to finalize choices
    if st.session_state.investment_asset_classes and st.session_state.borrowing_country and st.session_state.start_date and st.session_state.end_date:
        if st.button("Confirm Selections"):
            st.session_state.confirmed = True
            st.experimental_rerun()

    # Once confirmed, display the final table and print the selections
    if st.session_state.confirmed:
        st.markdown("<h3 style='text-align: center; color: green;'>Your Selections Have Been Confirmed!</h3>", unsafe_allow_html=True)
        st.write(f"Country to Borrow From: **{st.session_state.borrowing_country}**")
        st.write(f"Investment Period: {st.session_state.start_date} to {st.session_state.end_date}")

        # Display the selected assets
        st.write("### Final Investment Portfolio:")
        final_assets_df = pd.DataFrame(st.session_state.investment_asset_classes)
        final_assets_df["borrowing"]= st.session_state.borrowing_country
        final_assets_df["Start Date"]= st.session_state.start_date
        final_assets_df["End Date"]= st.session_state.end_date
        #final_assets_df.to_excel("scelta.xlsx")
        #st.write(final_assets_df)






        scelta = final_assets_df


        df = pd.read_excel("./fred_codes.xlsx",sheet_name="FRED codes")
        days90 = df[df['Name'].str.contains("3-Month or 90-Day ", case=False, na=False)]
        portfolio = pd.read_excel("./fred_codes.xlsx",sheet_name="Portfolio")
        stock = portfolio[["Country","Stock Index"]][portfolio['Stock Index']!=999]
        bond = portfolio[["Country","Bond"]][portfolio['Bond']!=999]
        reit = portfolio[["Country","REIT"]][portfolio['REIT']!=999]
        reit = portfolio[["Country","REIT"]][portfolio['REIT']!=999]
        choice = scelta.merge(days90[['Country', 'Currency']], left_on='country', right_on='Country', how='left').rename(columns={'Currency': 'invest_currency'})
        choice = choice.merge(days90[['Country', 'Currency']], left_on='borrowing', right_on='Country', how='left').rename(columns={'Currency': 'borrow_currency'})

        choice['exchange_pair'] = choice['invest_currency'] + choice['borrow_currency'] + "=X"


        choice['Start Date'] = pd.to_datetime(choice['Start Date'])
        choice['End Date'] = pd.to_datetime(choice['End Date'])


        my_start_date = choice['Start Date'].iloc[0].to_pydatetime()
        my_end_date = choice['End Date'].iloc[0].to_pydatetime()

        choice = choice.drop(columns=['Start Date', 'End Date',
                                    'invest_currency','borrow_currency',
                                    'Country_x','Country_y'])

        choice = choice.merge(days90[['Country', 'Code']], left_on='borrowing', right_on='Country', how='left').rename(columns={'Code': 'Borrowing Code'})
        choice['Investing Code'] = "-"
        choice['Investing Code'][0] =portfolio.loc[portfolio['Country'] == choice['country'][0], 'Stock Index'].values[0]
        choice['Investing Code'][1] =portfolio.loc[portfolio['Country'] == choice['country'][1], 'Bond'].values[0]
        choice['Investing Code'][2] =portfolio.loc[portfolio['Country'] == choice['country'][2], 'REIT'].values[0]
        choice['Label'] = choice['country'] + " - "+choice['asset_class']






        #if (my_start_date - my_end_date == datetime.timedelta(0)) or (my_end_date>date.today()):
        if (my_start_date.date() - my_end_date.date() == datetime.timedelta(0)) or (my_end_date.date() > date.today()):
            my_end_date = date.today()
            my_start_date = my_end_date - datetime.timedelta(365)
            
        rows_to_delete = []
        for index, row in choice.iterrows():
            if yf.download(f"{row['exchange_pair']}", start=my_start_date.strftime('%Y-%m-%d'), end=my_end_date.strftime('%Y-%m-%d'), interval='1d')['Close'].empty:
                rows_to_delete.append(index)
        choice = choice.drop(rows_to_delete)
        choice.reset_index(drop=True, inplace=True)
                
            
            
            
            

        fred_api_key = st.secrets["FRED_API_KEY"]
        fred = Fred(api_key=fred_api_key)
        DFS = []

        # FRED ####################################################################################
        data = {}

        data[choice['Borrowing Code'].iloc[0]] = fred.get_series(choice['Borrowing Code'].iloc[0])/100

        df = pd.DataFrame(data)
        df['Date'] = df.index
        #df['Date'] = df['Date'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])

        #if (df[(df['Date'] > pd.to_datetime(my_start_date)) & (df['Date'] < pd.to_datetime(my_end_date))].empty) or (my_end_date > date.today()):
        if (df[(df['Date'] > pd.to_datetime(my_start_date)) & (df['Date'] < pd.to_datetime(my_end_date))].empty) or (my_end_date.date() > date.today()):
            my_end_date = date.today()
            my_start_date = my_end_date - datetime.timedelta(365)
        
        #provaa   
        #my_end_date = date.today()
       #my_start_date = my_end_date - datetime.timedelta(365)

        df = df[(df['Date'] > pd.to_datetime(my_start_date)) & (df['Date'] < pd.to_datetime(my_end_date))].dropna()

        ###########################################################################################




        for index, row in choice.iterrows():

            data = yf.download([row['exchange_pair'],row['Investing Code']], start=my_start_date.strftime('%Y-%m-%d'), end=my_end_date.strftime('%Y-%m-%d'), interval='1d')['Close']
            
            data = data.reset_index()




            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            merged_df = pd.merge(data, df, on='Date', how='left')
            merged_df = merged_df.ffill()
            
            #merged_df[row['Investing Code']]= merged_df[row['Investing Code']].pct_change().shift(-1)
            

            
            
            merged_df['S_t+T'] = merged_df[row['exchange_pair']].shift(-1)
            

            merged = merged_df.dropna()


            # Drop the last row because S_{t+T} will be NaN for it
            merged_filtered = merged.dropna()
            merged_filtered = merged_filtered.reset_index(drop=True)
            T=90
            # Calculate the profit for each row
            merged_filtered['Profit %'] =  ((merged_filtered[row['Investing Code']]/merged_filtered[row['Investing Code']].iloc[0] - merged_filtered[row['Borrowing Code']]) * 

            (merged_filtered[row['exchange_pair']] / merged_filtered['S_t+T']))
            

             



            DFS.append(merged_filtered[['Date', 'Profit %']].rename(columns={'Profit %': row['Label']}))


            
            
            
        w_stock = choice.loc[choice['asset_class'] == "Stock Index", 'weight'].values[0]/100
        w_bond = choice.loc[choice['asset_class'] == "Bond", 'weight'].values[0]/100
        w_reit = choice.loc[choice['asset_class'] == "REIT", 'weight'].values[0]/100


        dfs = pd.concat(DFS, axis=1, join='inner')
        dfs = dfs.loc[:, ~dfs.columns.duplicated()]



        stock_column = [col for col in dfs.columns if "Stock" in col][0]
        bond_column = [col for col in dfs.columns if "Bond" in col][0]
        reit_column = [col for col in dfs.columns if "REIT" in col][0]

        dfs['Portfolio'] =  dfs[stock_column]*w_stock + dfs[bond_column]*w_bond + dfs[reit_column]*w_reit

        dfs.iloc[:, 1:] = dfs.iloc[:, 1:].applymap(lambda x: (x-1) * 100)  # Converting values to percentages


        dfs = dfs.iloc[::5]


        st.line_chart(
            dfs,
            x="Date",
            y=list(dfs.columns[1:]),
            color=generate_colors(len(dfs.columns[1:])),
            width=1200,  # Adjust height if needed
            height=800,  # Adjust height if needed
            use_container_width=True  # Use full width of the container
        )





    






# Main function to run both tabs
def main():
    
    menu = st.sidebar.selectbox("Select what you want to work with", ("Investment and Borrowing", "Asset Class Selection"))

    if menu == "Investment and Borrowing":
        try:
            create_pairs()
        except (IndexError, KeyError):
            reset_session()


    elif menu == "Asset Class Selection":
        try:
            create_investment_asset_selection()  # Call the function to handle investment and borrowing pairs
        except (IndexError, KeyError):
            reset_session()







    if st.button("Reset Choices"):
        reset_session()


main()

  # Re-inserting original content, including create_pairs()

# Display the formula at the end with LaTeX
st.markdown("""
<div class="formula-text">
    The profit from an investment can be calculated using the following formula:
</div>
""", unsafe_allow_html=True)

st.latex(r'''
P_{\text{profit}} = \text{Amount Invested} \times \left[ \left(1 + \left( i_{\text{domestic}} - i_{\text{foreign}} \right) \times \left(1 + \frac{S_t - S_{t+T}}{S_{t+T}}\right) \right)^{\frac{T}{360}} - 1 \right]
''')

# End of container
st.markdown('</div>', unsafe_allow_html=True)
