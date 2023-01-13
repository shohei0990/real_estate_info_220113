
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import japanize_matplotlib

plt.rcParams['font.family'] = 'MS Gothic'

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# csvファイルの読み込み
pre_df = pd.read_csv("realestateinfo_test_ishida91.csv") #pd.DataFrame(pre_data[1:], columns=col_name)  # 一段目をカラム、以下データフレームで取得
pre_df = pre_df.rename(columns={'面積': '面積[m^2]', '家賃': '家賃[万円]', '敷金': '敷金[万円]', '礼金': '礼金[万円]', '駅徒歩時間': '駅徒歩時間[分]'})
select_columns_num = ['面積[m^2]', '家賃[万円]', '敷金[万円]','礼金[万円]', '階数率', '築年数', '構造', '階数', '駅徒歩時間[分]']
pre0_df = pre_df[select_columns_num]

# object →　数字の変換 新しくスプシから取得する際には全てObjectになっているため。
for column in pre0_df:
    pre0_df[column] = pd.to_numeric(pre0_df[column], errors='coerce')

pre_df[select_columns_num] = pre0_df

# ---------------------------------------------------------------------------------------------------------------------------------------------- #

# 1. 画面の表示
# サイドバー
st.sidebar.title('散布表示の設定')
xmax = st.sidebar.number_input('【散布図】 x軸最大値：', 0, 1000, 100)
ymax = st.sidebar.number_input('【散布図】 y軸最大値：', 0, 1000, 200)

st.sidebar.title('ヒストグラム表示の設定')
bins2  = st.sidebar.number_input('【ヒスト】 ビン数：', 0, 500, 50)
xmax2  = st.sidebar.number_input('【ヒスト】 x軸最大値：', 0, 1000, 100)
ymax2  = st.sidebar.number_input('【ヒスト】 y軸最大値：', 0, 200000, 40000)
#slider('y軸最大値：', 0, 80000, 40000,1)

st.sidebar.title('箱ひげ図表示の設定')
ymax3  = st.sidebar.number_input('【箱ひげ】 y軸最大値：', 0, 1000, 40)
ymax4  = st.sidebar.number_input('【箱ひげ詳細】 y軸最大値：', 0, 1000, 40)

st.sidebar.title('その他設定')


# メイン
st.title('不動産情報の分析ページ1')

extra_configs_0 = st.expander("元のデータ表示") # Extra Configs
with extra_configs_0:
     st.write(pre_df.head(10))

extra_configs_1 = st.expander("基本統計量") # Extra Configs
with extra_configs_1:
     pre_df_dis = pre_df.describe()
     st.write(pre_df_dis)

# 相関関係
extra_configs_2 = st.expander("相関関係") # Extra Configs
with extra_configs_2:
    pre_df_corr = pre_df[['家賃[万円]', '面積[m^2]', '築年数','階数', '敷金[万円]', '礼金[万円]', '階数率', '駅徒歩時間[分]']].corr()
    st.write(pre_df_corr)

# ---------------------------------------------------------------------------------------------------------------------------------------------- #

st.markdown("---")
st.subheader('散布図(相関性)')
text_col, gra_col = st.columns([1,2], gap="medium")
# 2.横軸を実測値、縦軸を予測値として、散布図を描く
tx = text_col.selectbox('x軸値', ('家賃[万円]', '面積[m^2]', '築年数','階数', '敷金[万円]', '礼金[万円]', '階数率', '駅徒歩時間[分]') )
text_col.text ( '・'+ tx + 'の最大値:'+ str(pre_df[tx].max())  )
text_col.text ( '・'+ tx + 'の最小値:'+ str(pre_df[tx].min())  )
text_col.text ( '・'+ tx + 'の平均値:'+ str(pre_df[tx].mean()) )
ty = text_col.selectbox('y軸値', ('家賃[万円]', '面積[m^2]', '築年数','階数', '敷金[万円]', '礼金[万円]', '階数率', '駅徒歩時間[分]') )
text_col.text ( '・'+ ty + 'の最大値:'+ str(pre_df[tx].max())  )
text_col.text ( '・'+ ty + 'の最小値:'+ str(pre_df[tx].min())  )
text_col.text ( '・'+ ty + 'の平均値:'+ str(pre_df[tx].mean()) )

#group_color={"江戸川": 'red'}
fig = px.scatter(pre_df, x=tx, y=ty, 
                 range_x=[0, xmax],
                 range_y=[0, ymax],
                 hover_name="区",
                 #color="continent",
                 #color_discrete_sequence=px.colors.qualitative.Alphabet,
                 #color_discrete_map=group_color
                 )  # , hover_name="散布図"
#fig.update_layout(font_size=20, hoverlabel_font_size=20)
gra_col.plotly_chart(fig, use_container_width=True)  # , use_container_width=True

# ---------------------------------------------------------------------------------------------------------------------------------------------- #

st.markdown("---")
st.subheader('ヒストグラム')
text_col2, gra_col2 = st.columns([1,2], gap="medium")
tx2 = text_col2.selectbox('ヒスト対象', ('家賃[万円]', '面積[m^2]', '築年数','階数', '敷金[万円]', '礼金[万円]', '階数率', '駅徒歩時間[分]') )
text_col2.text ( '・'+ tx2 + 'の最大値:'+ str( pre_df[tx2].max() )  )
text_col2.text ( '・'+ tx2 + 'の最小値:'+ str( pre_df[tx2].min() )  )
text_col2.text ( '・'+ tx2 + 'の平均値:'+ str( pre_df[tx2].mean() ) )

fig  = plt.figure(figsize=(10,5))
plt.hist(pre_df[tx2], bins=bins2)
plt.xlim([0, xmax2])
plt.ylim([0, ymax2])
plt.title("ヒストグラム")
plt.xlabel(tx2)
plt.ylabel('カウント数')
gra_col2.pyplot(fig)


# 建物構造をx軸、家賃をy軸にした箱ひげ図を描いてください。??箱ひげ図は axが使えない
st.markdown("---")
st.subheader('箱ひげ図')
text_col3, gra_col3 = st.columns([1,3], gap="medium")
tx3 = text_col3.selectbox('分類対象', ('区','カテゴリー','間取り','市町', '最寄駅','路線') )
ty3 = text_col3.selectbox('Y軸対象', ('家賃[万円]', '面積[m^2]', '築年数','階数', '敷金[万円]', '礼金[万円]', '階数率', '駅徒歩時間[分]') )

fig3 = plt.figure(figsize=(12,6))
plt.xticks(rotation=80)
plt.ylim([0, ymax3])
plt.xlabel(tx3)
plt.ylabel(ty3)
plt.title("箱ひげ図")
sns.boxplot(x=tx3, y=ty3, data=pre_df)
gra_col3.pyplot(fig3)

st.markdown("---")
st.subheader('箱ひげ図_区詳細')
text_col4, gra_col4 = st.columns([1,3], gap="medium")
tx4 = text_col4.selectbox('区選択', ('品川','港','江戸川') )
tx5 = text_col4.selectbox('分類対象Ⅱ', ('カテゴリー','間取り','市町', '最寄駅','路線') )
ty6 = text_col4.selectbox('Y軸対象Ⅱ', ('家賃[万円]', '面積[m^2]', '築年数','階数', '敷金[万円]', '礼金[万円]', '階数率', '駅徒歩時間[分]') )

pre_df2 = pre_df[pre_df["区"]==tx4]

fig4 = plt.figure(figsize=(12,6))
plt.xticks(rotation=80)
plt.ylim([0, ymax4])
plt.xlabel(tx5)
plt.ylabel(ty6)
plt.title("箱ひげ図")
sns.boxplot(x=tx5, y=ty6, data=pre_df2)
gra_col4.pyplot(fig4)



