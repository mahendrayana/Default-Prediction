from IPython.display import display
import plotly.express as px

df = pd.read_csv('Data\\final_data.csv')
# Define your DataFrame df and other code here

# Create and display the first histogram figure
fig1 = px.histogram(df, x="emp_length_numeric", color='repay_fail', 
                    marginal="box", hover_data=df.columns)
display(fig1)

# Create and display the second histogram figure
fig2 = px.histogram(df, x="is_own_home", color='repay_fail', 
                    marginal="box", hover_data=df.columns)
display(fig2)

# Create and display the third histogram figure
fig3 = px.histogram(df, x="productive_prps", color='repay_fail', 
                    marginal="box", hover_data=df.columns)
display(fig3)

# Create and display the fourth histogram figure
fig4 = px.histogram(df, x="pymnt_progress", color='repay_fail', 
                    marginal="box", hover_data=df.columns)
display(fig4)

# Create and display the fifth histogram figure
fig5 = px.histogram(df, x="annual_inc", color='repay_fail', 
                    marginal="box", hover_data=df.columns)
display(fig5)
