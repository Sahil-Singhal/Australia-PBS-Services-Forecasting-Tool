# import all reqd. libraries
import pandas as pd
import numpy as np
import streamlit as st
import re
import matplotlib.pyplot as plt
plt.style.use('bmh')
import datetime as dt


st.title('PBS Services Forecast')

@st.cache_data()
def load_fact_data():
    return pd.read_csv('dos.csv')
fact = load_fact_data()

# group prescriptions by item code and month
rx_ic_m = fact.groupby(['MONTH_OF_SUPPLY', 'ITEM_CODE'], as_index=False)['PRESCRIPTIONS'].sum()

# load item code mapping
@st.cache_data()
def load_dim_ic_data():
    return pd.read_csv('pbs-item-drug-map.csv', encoding='latin-1')
dim_ic = load_dim_ic_data()

df = pd.merge(rx_ic_m, dim_ic, on='ITEM_CODE', how='inner')

@st.cache_data()
def load_streamlined_data():
    return pd.read_table('streamlined.txt', delimiter='\t')
streamlined = load_streamlined_data()

streamlined.loc[streamlined['mp-pt'].str.lower()=='adalimumab']['item-code'].count()

@st.cache_data()
def load_restriction_data():
    return pd.read_table('RestrictionExtractDelimited.txt', delimiter='\t')
restriction = load_restriction_data()

def find_condition(string):
    pattern = r".[Tt]reatment [Pp]hase.*|.[Cc]linical [Cc]riteria.|.[Pp]opulation [Cc]riteria.|.[Tt]reatment [Cc]riteria."
    match = re.split(pattern, string)
    if match:
        return match[0]
    else:
        return "n/a"

restriction['condition'] = restriction['restriction-text'].apply(find_condition)
restriction = restriction[['treatment-of-code', 'condition']].drop_duplicates()

condition = pd.merge(streamlined, restriction, on='treatment-of-code')

condition = condition[['item-code', 'condition']].drop_duplicates()

condition = condition.groupby(['item-code'], as_index=False)['condition'].apply(lambda x: ','.join(x))

df = pd.merge(df, condition, left_on='ITEM_CODE', right_on='item-code', how='left')


df['MONTH_OF_SUPPLY'] = pd.to_datetime(df.MONTH_OF_SUPPLY, format="%Y%m")
#df['MONTH_OF_SUPPLY'] = df['MONTH_OF_SUPPLY'].dt.strftime('%b-%y')

#"""**PBS prescription trend**"""

list_of_drugs = np.sort(df.DRUG_NAME.unique())

# streamlit dropdown
drug_name = st.sidebar.multiselect("Drug name", list_of_drugs, default='ABEMACICLIB')
st.sidebar.write('##')
lbp = st.sidebar.slider(label='Adjust lookback period to base forecast on', min_value=6, max_value=48, step=6, value=12)


if not(drug_name): 
    st.text('Select a drug(s) from the dropdown on the left')
else:
    names = ', '.join(drug_name)
    st.markdown('*Showing trends for* ' + ':red[' + names +']')
    st.write('#')

    df2 = df[df.DRUG_NAME.isin(drug_name)]

    df2 = df2.groupby('MONTH_OF_SUPPLY', as_index=False)['PRESCRIPTIONS'].sum()

    df2.sort_values(by='MONTH_OF_SUPPLY', inplace=True)


    # log regression
    df_np = df2.to_numpy()

    # x = np.array([i for i in range(len(df_np)+1)[1:]], dtype='float')
    x = df2.index + 1
    y = df2['PRESCRIPTIONS']
    fit = np.polyfit(np.log(x[-lbp:]), y[-lbp:], 1)

    last_mth_of_actuals = df2['MONTH_OF_SUPPLY'].max()
    fcst_end_yr = dt.datetime.today().year + 2
    fcst_till_period = dt.date(fcst_end_yr, 12, 1)
    fcst_date_range = pd.date_range(start=last_mth_of_actuals, end=fcst_till_period, freq='MS', closed='right')

    x_fcst = np.array([i for i in range(len(x)+1, len(x) + 1 + len(fcst_date_range))], dtype='float')
    y_fcst = np.round(np.log(x_fcst) * fit[0] + fit[1],0)

    # print(y)
    # print(y_fcst)

    final_x = np.hstack((df2['MONTH_OF_SUPPLY'],fcst_date_range))
    final_y = np.hstack((y,y_fcst))

    col1, col2, col3, col4 = st.columns(4)

    # metric for last 12m growth
    y_l12m = sum(y[-12:])
    y_p12m = sum(y[-24:-12])
    gr = y_l12m / y_p12m -1
    g_l12m = str(round(gr*100,1)) + '%'
    with col1:
        st.metric(value = g_l12m, label='Past 12M Growth')

    # metric for next 12m growth 
    y_n12m = sum(y_fcst[:12])
    gr_n12m = y_n12m / y_l12m -1
    g_n12m = str(round(gr_n12m*100,1)) + '%'
    with col2: 
        st.metric(value = g_n12m, label='Next 12M Growth')

    final_df = pd.DataFrame(final_y, final_x, columns=['Prescriptions'])

    # metric for next calendar year growth
    ncy = dt.datetime.now().year + 1
    rx_ncy = final_df.loc[final_df.index.year == ncy].sum()
    rx_ccy = final_df.loc[final_df.index.year == dt.datetime.now().year].sum()
    gr_ncy = str(round(100 * (rx_ncy / rx_ccy - 1),1)[0]) + '%'

    pcy = dt.datetime.now().year -1
    rx_pcy = final_df.loc[final_df.index.year == pcy].sum()
    gr_ccy = str(round(100 * (rx_ccy / rx_pcy - 1),1)[0]) + '%'

    with col3:
        st.metric(value = gr_ccy, label=str(ncy-1) + ' Growth')
    with col4:
        st.metric(value = gr_ncy, label=str(ncy) + ' Growth')

    st.write('#')

    # actuals and forecast lines separated
    fcst_x = final_x[final_x>last_mth_of_actuals]

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(df2['MONTH_OF_SUPPLY'], y, color='black', label='Actuals')
    ax.plot(fcst_x, y_fcst, ls=':', color='grey', label='Forecast')
    ax.legend()
    ax.set_title('PRESCRIPTION TREND OVER TIME', size=10)

    st.pyplot(plt)

    st.markdown('#')
    st.markdown('#### NOTE')
    st.markdown('- The forecast is based on the past trend of prescriptions.')
    st.markdown('- It is assumed that the past trend will continue.')
    st.markdown('- If you expect changes in the market or competition, then feel free to download the chart data and make adjustments accordingly.')
    st.write('#')
    st.download_button(
        label="Download data",
        data=pd.DataFrame(final_y, final_x, columns=['Prescriptions']).transpose().to_csv(),
        file_name="chart_data.csv",
        mime="text/csv",
    )
