
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import japanize_matplotlib

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

japanize_matplotlib.japanize() 
plt.rcParams['font.family'] = 'MS Gothic'

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="ð§",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# csvãã¡ã¤ã«ã®èª­ã¿è¾¼ã¿
pre_df = pd.read_csv("realestateinfo_test2_ishida91.csv") #pd.DataFrame(pre_data[1:], columns=col_name)  # ä¸æ®µç®ãã«ã©ã ãä»¥ä¸ãã¼ã¿ãã¬ã¼ã ã§åå¾
pre_df = pre_df.rename(columns={'é¢ç©': 'é¢ç©[m^2]', 'å®¶è³': 'å®¶è³[ä¸å]', 'æ·é': 'æ·é[ä¸å]', 'ç¤¼é': 'ç¤¼é[ä¸å]', 'é§å¾æ­©æé': 'é§å¾æ­©æé[å]'})
select_columns_num = ['é¢ç©[m^2]', 'å®¶è³[ä¸å]', 'æ·é[ä¸å]','ç¤¼é[ä¸å]', 'éæ°ç', 'ç¯å¹´æ°', 'æ§é ', 'éæ°', 'é§å¾æ­©æé[å]']
pre0_df = pre_df[select_columns_num]

# object âãæ°å­ã®å¤æ æ°ããã¹ãã·ããåå¾ããéã«ã¯å¨ã¦Objectã«ãªã£ã¦ããããã
for column in pre0_df:
    pre0_df[column] = pd.to_numeric(pre0_df[column], errors='coerce')

pre_df[select_columns_num] = pre0_df

# ---------------------------------------------------------------------------------------------------------------------------------------------- #

# 1. ç»é¢ã®è¡¨ç¤º
# ãµã¤ããã¼
st.sidebar.title('æ£å¸è¡¨ç¤ºã®è¨­å®')
xmax = st.sidebar.number_input('ãæ£å¸å³ã xè»¸æå¤§å¤ï¼', 0, 1000, 100)
ymax = st.sidebar.number_input('ãæ£å¸å³ã yè»¸æå¤§å¤ï¼', 0, 1000, 200)

st.sidebar.title('ãã¹ãã°ã©ã è¡¨ç¤ºã®è¨­å®')
bins2  = st.sidebar.number_input('ããã¹ãã ãã³æ°ï¼', 0, 500, 100)
xmax2  = st.sidebar.number_input('ããã¹ãã xè»¸æå¤§å¤ï¼', 0, 1000, 100)
ymax2  = st.sidebar.number_input('ããã¹ãã yè»¸æå¤§å¤ï¼', 0, 200000, 40000)
#slider('yè»¸æå¤§å¤ï¼', 0, 80000, 40000,1)

st.sidebar.title('ç®±ã²ãå³è¡¨ç¤ºã®è¨­å®')
ymax3  = st.sidebar.number_input('ãç®±ã²ãã yè»¸æå¤§å¤ï¼', 0, 1000, 40)
ymax4  = st.sidebar.number_input('ãç®±ã²ãè©³ç´°ã yè»¸æå¤§å¤ï¼', 0, 1000, 40)

st.sidebar.title('å®¶è³äºæ¸¬ã®è¨­å®')
val5   = st.sidebar.number_input('ãå®¶è³äºæ¸¬ã ä¹±æ°ã·ã¼ãæå®ï¼', 0, 200, 42)
ymax5  = st.sidebar.number_input('ãå®¶è³äºæ¸¬ã xyè»¸æå¤§å¤ï¼'   , 0, 1000, 50)


# ã¡ã¤ã³
st.title('æ±äº¬é½ã®ä¸åç£æå ±ã®åæãã¼ã¸')

extra_configs_0 = st.expander("åã®ãã¼ã¿è¡¨ç¤º") # Extra Configs
with extra_configs_0:
     st.write(pre_df)

extra_configs_1 = st.expander("åºæ¬çµ±è¨é") # Extra Configs
with extra_configs_1:
     pre_df_dis = pre_df.describe()
     st.write(pre_df_dis)

# ç¸é¢é¢ä¿
extra_configs_2 = st.expander("ç¸é¢é¢ä¿") # Extra Configs
with extra_configs_2:
    pre_df_corr = pre_df[['å®¶è³[ä¸å]', 'é¢ç©[m^2]', 'ç¯å¹´æ°','éæ°', 'æ·é[ä¸å]', 'ç¤¼é[ä¸å]', 'éæ°ç', 'é§å¾æ­©æé[å]']].corr()
    st.write(pre_df_corr)

# ---------------------------------------------------------------------------------------------------------------------------------------------- #
japanize_matplotlib.japanize() 
st.markdown("---")
st.subheader('æ£å¸å³(ç¸é¢æ§)')
text_col, gra_col = st.columns([1,2], gap="medium")
# 2.æ¨ªè»¸ãå®æ¸¬å¤ãç¸¦è»¸ãäºæ¸¬å¤ã¨ãã¦ãæ£å¸å³ãæã
tx = text_col.selectbox('xè»¸å¤', ('å®¶è³[ä¸å]', 'é¢ç©[m^2]', 'ç¯å¹´æ°','éæ°', 'æ·é[ä¸å]', 'ç¤¼é[ä¸å]', 'éæ°ç', 'é§å¾æ­©æé[å]') )
text_col.text ( 'ã»'+ tx + 'ã®æå¤§å¤:'+ str(pre_df[tx].max())  )
text_col.text ( 'ã»'+ tx + 'ã®æå°å¤:'+ str(pre_df[tx].min())  )
text_col.text ( 'ã»'+ tx + 'ã®å¹³åå¤:'+ str(pre_df[tx].mean()) )
ty = text_col.selectbox('yè»¸å¤', ( 'é¢ç©[m^2]','å®¶è³[ä¸å]', 'ç¯å¹´æ°','éæ°', 'æ·é[ä¸å]', 'ç¤¼é[ä¸å]', 'éæ°ç', 'é§å¾æ­©æé[å]') )
text_col.text ( 'ã»'+ ty + 'ã®æå¤§å¤:'+ str(pre_df[tx].max())  )
text_col.text ( 'ã»'+ ty + 'ã®æå°å¤:'+ str(pre_df[tx].min())  )
text_col.text ( 'ã»'+ ty + 'ã®å¹³åå¤:'+ str(pre_df[tx].mean()) )

#group_color={"æ±æ¸å·": 'red'}
fig = px.scatter(pre_df, x=tx, y=ty, 
                 range_x=[0, xmax],
                 range_y=[0, ymax],
                 hover_name="åº",
                 #color="continent",
                 #color_discrete_sequence=px.colors.qualitative.Alphabet,
                 #color_discrete_map=group_color
                 )  # , hover_name="æ£å¸å³"
#fig.update_layout(font_size=20, hoverlabel_font_size=20)
gra_col.plotly_chart(fig, use_container_width=True)  # , use_container_width=True

# ---------------------------------------------------------------------------------------------------------------------------------------------- #

st.markdown("---")
st.subheader('ãã¹ãã°ã©ã ')
text_col2, gra_col2 = st.columns([1,2], gap="medium")
tx2 = text_col2.selectbox('ãã¹ãå¯¾è±¡', ('å®¶è³[ä¸å]', 'é¢ç©[m^2]', 'ç¯å¹´æ°','éæ°', 'æ·é[ä¸å]', 'ç¤¼é[ä¸å]', 'éæ°ç', 'é§å¾æ­©æé[å]') )
text_col2.text ( 'ã»'+ tx2 + 'ã®æå¤§å¤:'+ str( pre_df[tx2].max() )  )
text_col2.text ( 'ã»'+ tx2 + 'ã®æå°å¤:'+ str( pre_df[tx2].min() )  )
text_col2.text ( 'ã»'+ tx2 + 'ã®å¹³åå¤:'+ str( pre_df[tx2].mean() ) )

fig  = plt.figure(figsize=(10,5))
plt.hist(pre_df[tx2], bins=bins2)
plt.xlim([0, xmax2])
plt.ylim([0, ymax2])
plt.title("ãã¹ãã°ã©ã ")
plt.xlabel(tx2)
plt.ylabel('ã«ã¦ã³ãæ°')
gra_col2.pyplot(fig)


# å»ºç©æ§é ãxè»¸ãå®¶è³ãyè»¸ã«ããç®±ã²ãå³ãæãã¦ãã ããã??ç®±ã²ãå³ã¯ axãä½¿ããªã
st.markdown("---")
st.subheader('ç®±ã²ãå³')
text_col3, gra_col3 = st.columns([1,3], gap="medium")
tx3 = text_col3.selectbox('åé¡å¯¾è±¡', ('åº','ã«ãã´ãªã¼','éåã','å¸çº', 'æå¯é§','è·¯ç·') )
ty3 = text_col3.selectbox('Yè»¸å¯¾è±¡', ('å®¶è³[ä¸å]', 'é¢ç©[m^2]', 'ç¯å¹´æ°','éæ°', 'æ·é[ä¸å]', 'ç¤¼é[ä¸å]', 'éæ°ç', 'é§å¾æ­©æé[å]') )

fig3 = plt.figure(figsize=(12,6))
plt.xticks(rotation=80)
plt.ylim([0, ymax3])
plt.xlabel(tx3)
plt.ylabel(ty3)
plt.title("ç®±ã²ãå³")
sns.boxplot(x=tx3, y=ty3, data=pre_df)
gra_col3.pyplot(fig3)

st.markdown("---")
st.subheader('ç®±ã²ãå³_åºè©³ç´°')
text_col4, gra_col4 = st.columns([1,3], gap="medium")
tx4 = text_col4.selectbox('åºé¸æ', ('åå·','æ¸¯','æ±æ¸å·') )
tx5 = text_col4.selectbox('åé¡å¯¾è±¡â¡', ('ã«ãã´ãªã¼','éåã','å¸çº', 'æå¯é§','è·¯ç·') )
ty6 = text_col4.selectbox('Yè»¸å¯¾è±¡â¡', ('å®¶è³[ä¸å]', 'é¢ç©[m^2]', 'ç¯å¹´æ°','éæ°', 'æ·é[ä¸å]', 'ç¤¼é[ä¸å]', 'éæ°ç', 'é§å¾æ­©æé[å]') )

pre_df2 = pre_df[pre_df["åº"]==tx4]

fig4 = plt.figure(figsize=(12,6))
plt.xticks(rotation=80)
plt.ylim([0, ymax4])
plt.xlabel(tx5)
plt.ylabel(ty6)
plt.title("ç®±ã²ãå³")
sns.boxplot(x=tx5, y=ty6, data=pre_df2)
gra_col4.pyplot(fig4)

#-----------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader('ã©ã³ãã ãã©ã¬ã¹ãã«ããå®¶è³æ¨å®')
tx10 = st.multiselect('å­¦ç¿ç¨ãã¼ã¿', ['é¢ç©[m^2]','ç¯å¹´æ°','éæ°','éæ°ç','é§å¾æ­©æé[å]','åº','éåã','å¸çº'], ['é¢ç©[m^2]','ç¯å¹´æ°']) 
select_columns5 = tx10 

text_col5, gra_col5 = st.columns([1,3], gap="medium")

if text_col5.button('äºæ¸¬éå§'):
    # å­¦ç¿ã¾ã§ã®ãã¼ã¿æºå
    data0 = pre_df[:60000] # å­¦ç¿ç¨
    test0 = pre_df[60000:] # æ¯è¼ç¨
    data0[select_columns5] = data0[select_columns5].fillna(-99)
    test0[select_columns5] = test0[select_columns5].fillna(-99)
    X_train, y_train = pd.get_dummies(data0[select_columns5]), data0['å®¶è³[ä¸å]']
    X_test , y_test  = pd.get_dummies(test0[select_columns5]), test0['å®¶è³[ä¸å]']

    rf = RandomForestRegressor(random_state=val5)
    rf.fit(X_train, y_train)

    # è©ä¾¡ãã¼ã¿ã«å¯¾ããäºæ¸¬ãè¡ãããã®çµæãå¤æ°predã«ä»£å¥ãã¦ãã ããã
    pred = rf.predict(X_test)

    # äºæ¸¬ç²¾åº¦ã®ç¢ºèª
    new = np.sqrt(MSE(y_test, pred))
    text_col5.text ( "ã»RMSEï¼"+ str(new)  )

    # ã°ã©ãã®å¯è¦å
    fig5 = px.scatter(x=y_test, y=pred, 
                      title="xè»¸ï¼å®¶è³[ä¸å]ãVSããyè»¸ï¼äºæ¸¬å®¶è³[ä¸å]",
                      range_x=[0, ymax5],
                      range_y=[0, ymax5],
                 )    
    gra_col5.plotly_chart(fig5, use_container_width=True) 

else:
    st.write('äºæ¸¬ã»æ¯è¼ã°ã©ããåºãã«ã¯ããã¿ã³ãæ¼ãã¦ã­ã')


