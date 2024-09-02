#!/usr/bin/env python
# coding: utf-8

# # End-to-End Cricket Data Analysis
# 
# ### This project showcases the use of SQL, Python to analyze Cricket Players Performance stats for building best T20 world team. It also shows how to use PANEL library from Holoviz to build a data visualization app. 
# 

# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Data Collection</span>

# In[5]:


import mysql.connector


# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# import numpy as np

# In[9]:


from bokeh.models.formatters import NumeralTickFormatter
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv
from panel.interact import interact
pn.extension('tabulator')
pn.extension(loading_spinner = 'petal', loading_color = '#00aa41')
import param
from PIL import Image
import io
import base64
import hvplot.pandas
from panel.template import GoldenTemplate
from panel.theme import DarkTheme

formatter = NumeralTickFormatter(format='0.0,a')


# ## Below code shows how we can pull data from MYSQL. 
# ## For deployment to Huggingface, we have exported the combined dataframe and will be using the data going further.

# In[10]:


# conn = mysql.connector.connect(
# host ="localhost",
# user = "root",
# password = "root",
# database = "cricket_dataproject")

# mycursor = conn.cursor()
# mycursor.execute("SELECT * FROM finalbatting")
# myresult = mycursor.fetchall()
# batting = pd.DataFrame(myresult, columns=[desc[0] for desc in mycursor.description])
# mycursor.close()
# conn.close()

# conn = mysql.connector.connect(
# host ="localhost",
# user = "root",
# password = "root",
# database = "cricket_dataproject")

# mycursor = conn.cursor()
# mycursor.execute("SELECT * FROM bowlinginfo2")
# myresult = mycursor.fetchall()
# bowling = pd.DataFrame(myresult, columns=[desc[0] for desc in mycursor.description])
# mycursor.close()
# conn.close()

# # conn = mysql.connector.connect(
# # host ="localhost",
# # user = "root",
# # password = "root",
# # database = "cricket_dataproject")

# # mycursor = conn.cursor()
# # mycursor.execute("SELECT * FROM dim_players")
# # myresult = mycursor.fetchall()
# # players = pd.DataFrame(myresult, columns=[desc[0] for desc in mycursor.description])
# # mycursor.close()
# # conn.close()

# Combined = batting.merge(bowling , on="Name", how="outer")
# Combined.fillna(0, inplace=True)

# # Convert specific columns
# columns_to_convert = ['Total_Runs', 'Total_Innings_x', 'Total_balls_faced','Avg_Batting_Pos','Avg_balls_faced','Total_boundary_runs',
#                       'Strike_rate','Bound_percent','innings_dismissed','Batting_Avg','Total_wickets','Total_balls','Total_runs_conceded',
#                       'Total_Innings_y','Ballsnoruns','Bowling_Economy','Bowling_strikerate','Bowling_Average','DotballPercent']
# Combined[columns_to_convert] = Combined[columns_to_convert].astype(int)
# Combined.rename(columns={'Total_Innings_x':'Total_Innings_bat','Total_Innings_y':'Total_Innings_bowl'}, inplace=True)

# Combined.to_excel("FinalDatafromSQLqueries.xlsx")


# In[11]:


Combined = pd.read_excel("./FinalDatafromSQLqueries.xlsx")


# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Data Filtering to find best players</span>

# ## Cricket team is made of 11 players. The goal of these 11 players to score more runs if they are batting second or restrict other team to score less than them if they batted first
# 
# ## 11 players are basically divided into 4 categories
# 
# ### Opening batsman - Face best bowlers and the new ball which moves a lot 
# ### Higher order Anchors batsman - if opening batsman are out quickly they can anchor innings
# ### Lower Order Anchor players - In the middle overs, when restrictions are there in the field, keep moving the scorecard at decent pace
# ### All-Rounders - Can do everything - Increase score rapidly or take wickets when bowling
# ### Best Bowlers - Take wickets without giving away too many runs

# In[12]:


def filter_dataframe(df, filter_conditions):
    """
    Filter a pandas DataFrame based on multiple conditions for multiple columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to filter
    filter_conditions (dict): A dictionary where keys are column names and values are lists of conditions
    
    Returns:
    pd.DataFrame: A new DataFrame containing only the rows that meet all specified conditions
    """
    mask = pd.Series(True, index=df.index)
    
    for column, conditions in filter_conditions.items():
        column_mask = pd.Series(False, index=df.index)
        for condition in conditions:
            if isinstance(condition, (list, tuple)) and len(condition) == 2:
                operator, value = condition
                if operator == '==':
                    column_mask |= df[column] == value
                elif operator == '!=':
                    column_mask |= df[column] != value
                elif operator == '>':
                    column_mask |= df[column] > value
                elif operator == '>=':
                    column_mask |= df[column] >= value
                elif operator == '<':
                    column_mask |= df[column] < value
                elif operator == '<=':
                    column_mask |= df[column] <= value
                elif operator == 'in':
                    column_mask |= df[column].isin(value)
                elif operator == 'not in':
                    column_mask |= ~df[column].isin(value)
            else:
                column_mask |= df[column] == condition
        
        mask &= column_mask
    
    return df[mask]


# In[13]:


opener_conditions = {
    'Batting_Avg': [('>', 30)],  # 2 < A <= 4
    'Strike_rate': [('>', 140)],             # B is 'x' or 'y'
    'Total_Innings_bat': [('>', 3)],       # C >= 20
    'Bound_percent': [('>', 50)],
    'Avg_Batting_Pos': [('<', 4)]
}

Openers = filter_dataframe(Combined, opener_conditions)


# In[14]:


Openers1 = Openers[["Name","Batting_Avg","Strike_rate","Bound_percent"]]


# In[15]:


HAnchors_conditions = {
    'Batting_Avg': [('>', 40)],  # 2 < A <= 4
    'Strike_rate': [('>', 125)],             # B is 'x' or 'y'
    'Total_Innings_bat': [('>', 3)],       # C >= 20
    'Avg_balls_faced': [('>', 20)],
    'Avg_Batting_Pos': [('>', 2)]
}

HAnchors = filter_dataframe(Combined, HAnchors_conditions)


# In[16]:


LAnchors_conditions = {
    'Batting_Avg': [('>', 20)],  # 2 < A <= 4
    'Strike_rate': [('>', 130)],             # B is 'x' or 'y'
    'Total_Innings_bat': [('>', 3)],       # C >= 20
    'Avg_balls_faced': [('>', 12)],
    'Avg_Batting_Pos': [('>', 4)],
    'Total_Innings_bowl': [('>', 1)]
}

LAnchors = filter_dataframe(Combined, LAnchors_conditions)


# In[17]:


Allrounders_conditions = {
    'Batting_Avg': [('>', 15)],  # 2 < A <= 4
    'Strike_rate': [('>', 140)],             # B is 'x' or 'y'
    'Total_Innings_bat': [('>', 2)],       # C >= 20
    'Avg_Batting_Pos': [('>', 4)],
    'Total_Innings_bowl': [('>', 2)],
    'Bowling_Economy': [('<', 7)],
    'Bowling_strikerate': [('<', 20)]
}

Allrounders = filter_dataframe(Combined, Allrounders_conditions)


# In[18]:


bowler_conditions = {
    'Total_Innings_bowl': [('>', 4)],  # 2 < A <= 4
    'Bowling_Economy': [('<', 7)],             # B is 'x' or 'y'
    'Bowling_strikerate': [('<', 16)],       # C >= 20
    'Bowling_Average': [('>', 4)],
    'DotballPercent': [('>', 40)]
}

bowlers = filter_dataframe(Combined, bowler_conditions)


# In[19]:


Openers.loc[Openers.Name !='',"Category"] = "Opener"
HAnchors.loc[HAnchors.Name !='',"Category"] = "HAnchors"
LAnchors.loc[LAnchors.Name !='',"Category"] = "LAnchors"
Allrounders.loc[Allrounders.Name !='',"Category"] = "Allrounders"
bowlers.loc[bowlers.Name !='',"Category"] = "bowlers"


# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Data setup for interactive dashboard</span>

# In[20]:


Openers_list = Openers['Name'].unique().tolist()
HAnchors_list = HAnchors['Name'].unique().tolist()
LAnchors_list = LAnchors['Name'].unique().tolist()
Allrounders_list = Allrounders['Name'].unique().tolist()
bowlers_list = bowlers['Name'].unique().tolist()


# ## Using css for styling

# In[21]:


css_sample = {
    'background-color' : '#5F9EA0',
    'border': '2px solid black',
    'color':'black',
    'padding':'15px 20px',
    'text-align':'center',
    'text-decoration':'none',
    'font-size':'20px',
    'font-family':'tahoma',
    'margin':'10px 50px',
    'cursor':'move'
}

css_sample2 = {
    'background-color' : '#5F9EA0',
    'border': '',
    'color':'black',
    'padding':'5px 10px',
    'text-align':'center',
    'text-decoration':'none',
    'font-family':'tahoma',
    'margin':'10px 50px',
    'cursor':'move'
}

custom_css = """
<style>
.bk-root {
    font-family: 'Roboto', sans-serif;
}
.app-header {
    font-size: 24px;
    font-weight: 300;
    color: #333;
    margin-bottom: 20px;
}
.image-column {
    display: inline-block;
    vertical-align: top;
    margin-right: 20px;
    margin-bottom: 20px;
}
.image-container {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}
.image-container:hover {
    transform: translateY(-5px);
}
.image-description {
    font-size: 14px;
    color: #666;
    margin-top: 10px;
}
</style>
"""


# ### Designing Panel widgets

# In[22]:


Select_Opener = pn.widgets.MultiSelect(name = 'Select_2_Openers', options = Openers_list,min_width=75 ,max_width = 300, min_height = 100, max_height=200, value = Openers_list, styles = css_sample, size=7 ) 
Select_HAnchors = pn.widgets.MultiSelect(name = 'Select_MiddleOrder', options = HAnchors_list,min_width=75 ,max_width = 300, value = HAnchors_list, styles = css_sample ) 
Select_LAnchors = pn.widgets.MultiSelect(name = 'Select_LowerMiddleOrder', options = LAnchors_list,min_width=75 ,max_width = 300, value = LAnchors_list, styles = css_sample ) 
Select_Allrounders = pn.widgets.MultiSelect(name = 'Select_Allrounders', options = Allrounders_list,min_width=75 ,max_width = 300, value = Allrounders_list, styles = css_sample ) 
Select_bowlers = pn.widgets.MultiSelect(name = 'Select_bowlers', options = bowlers_list,min_width=75 ,max_width = 300, value = bowlers_list, styles = css_sample ) 


# ## For the interactive dashboard,
# ## as we select player names to see combined performance,
# ## I also want to see their pictures. We have developed a class for each of the categories

# In[23]:


class ImageSelector(param.Parameterized):
    def __init__(self, Select_Opener, **params):
        super().__init__(**params)
        self.Select_Opener = Select_Opener
        self.image_dict = {
            'Alex Hales': {'path': '.\Images\Alex_Hales.jpg', 'description': 'Alex Hales'},
            'Jos Buttler(c)': {'path': '.\Images\Jos_Buttler(c).jpg', 'description': 'Jos Buttler(c)'},
            'Kusal Mendis': {'path': '.\Images\Kusal_Mendis.jpg', 'description': 'Kusal Mendis'},
            'Quinton de Kock': {'path': '.\Images\Quinton_de_Kock.jpg', 'description': 'Quinton de Kock'},
            'Rilee Rossouw': {'path': '.\Images\Rilee_Rossouw.jpg', 'description': 'Rilee Rossouw'}
        }

    @param.depends('Select_Opener.value')
    def view(self):
        if not self.Select_Opener.value:
            return pn.pane.Markdown("Please select 2 of the opener options.")

        images = []
        for animal in self.Select_Opener.value:
            img = Image.open(self.image_dict[animal]['path'])
            img.thumbnail((200, 200))  # Resize image to a maximum of 200x200
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            images.append(
                pn.Column(
                    pn.pane.HTML(f'''
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}">
                        </div>
                    '''),
                    pn.pane.Markdown(f"**{animal}:** {self.image_dict[animal]['description']}",
                                     css_classes=['image-description']),
                    css_classes=['image-column']
                )
            )

        return pn.Row(*images)
    
selector = ImageSelector(Select_Opener)


# In[24]:


class HAnchorImageSelector(param.Parameterized):
    def __init__(self, Select_HAnchors, **params):
        super().__init__(**params)
        self.Select_HAnchors = Select_HAnchors
        self.image_dict = {
            'Daryl Mitchell': {'path': '.\Images\Daryl_Mitchell.jpg', 'description': 'Daryl_Mitchell'},
            'Suryakumar Yadav': {'path': '.\Images\Suryakumar_Yadav.jpg', 'description': 'Suryakumar_Yadav'},
            'Virat Kohli': {'path': '.\Images\Virat_Kohli.jpg', 'description': 'Virat_Kohli'}
        }

    @param.depends('Select_HAnchors.value')
    def view(self):
        if not self.Select_HAnchors.value:
            return pn.pane.Markdown("Please select 2 of the opener options.")

        images = []
        for animal in self.Select_HAnchors.value:
            img = Image.open(self.image_dict[animal]['path'])
            img.thumbnail((200, 200))  # Resize image to a maximum of 200x200
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            images.append(
                pn.Column(
                    pn.pane.HTML(f'''
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}">
                        </div>
                    '''),
                    pn.pane.Markdown(f"**{animal}:** {self.image_dict[animal]['description']}",
                                     css_classes=['image-description']),
                    css_classes=['image-column']
                )
            )

        return pn.Row(*images)
    
Hanchorselector = HAnchorImageSelector(Select_HAnchors)


# In[25]:


class LAnchorImageSelector(param.Parameterized):
    def __init__(self, Select_LAnchors, **params):
        super().__init__(**params)
        self.Select_LAnchors = Select_LAnchors
        self.image_dict = {
            'Curtis Campher': {'path': '.\Images\Curtis_Campher.jpg', 'description': 'Curtis_Campher'},
            'Glenn Maxwell': {'path': '.\Images\Glenn_Maxwell.jpg', 'description': 'Glenn_Maxwell'},
            'Hardik Pandya': {'path': '.\Images\Hardik_Pandya.jpg', 'description': 'Hardik_Pandya'},
            'Marcus Stoinis': {'path': '.\Images\Marcus_Stoinis.jpg', 'description': 'Marcus_Stoinis'},
            'Sikandar Raza': {'path': '.\Images\Sikandar_Raza.jpg', 'description': 'Sikandar_Raza'}
        }

    @param.depends('Select_LAnchors.value')
    def view(self):
        if not self.Select_LAnchors.value:
            return pn.pane.Markdown("Please select 2 of the opener options.")

        images = []
        for animal in self.Select_LAnchors.value:
            img = Image.open(self.image_dict[animal]['path'])
            img.thumbnail((200, 200))  # Resize image to a maximum of 200x200
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            images.append(
                pn.Column(
                    pn.pane.HTML(f'''
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}">
                        </div>
                    '''),
                    pn.pane.Markdown(f"**{animal}:** {self.image_dict[animal]['description']}",
                                     css_classes=['image-description']),
                    css_classes=['image-column']
                )
            )

        return pn.Row(*images)
    
LAnchorselector = LAnchorImageSelector(Select_LAnchors)


# In[26]:


class AllRoundersImageSelector(param.Parameterized):
    def __init__(self, Select_Allrounders, **params):
        super().__init__(**params)
        self.Select_Allrounders = Select_Allrounders
        self.image_dict = {
            'Mitchell Santner': {'path': '.\Images\Mitchell_Santner.jpg', 'description': 'Mitchell_Santner'},
            'Rashid Khan': {'path': '.\Images\Rashid_Khan.jpg', 'description': 'Rashid_Khan'},
            'Shadab Khan': {'path': '.\Images\Shadab_Khan.jpg', 'description': 'Shadab_Khan'}
        }

    @param.depends('Select_Allrounders.value')
    def view(self):
        if not self.Select_Allrounders.value:
            return pn.pane.Markdown("Please select 2 of the opener options.")

        images = []
        for animal in self.Select_Allrounders.value:
            img = Image.open(self.image_dict[animal]['path'])
            img.thumbnail((200, 200))  # Resize image to a maximum of 200x200
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            images.append(
                pn.Column(
                    pn.pane.HTML(f'''
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}">
                        </div>
                    '''),
                    pn.pane.Markdown(f"**{animal}:** {self.image_dict[animal]['description']}",
                                     css_classes=['image-description']),
                    css_classes=['image-column']
                )
            )

        return pn.Row(*images)
    
AllRounderselector = AllRoundersImageSelector(Select_Allrounders)


# In[27]:


class BowlersImageSelector(param.Parameterized):
    def __init__(self, Select_bowlers, **params):
        super().__init__(**params)
        self.Select_bowlers = Select_bowlers
        self.image_dict = {
            'Anrich Nortje': {'path': '.\Images\Anrich_Nortje.jpg', 'description': 'Anrich_Nortje'},
            'Mitchell Santner': {'path': '.\Images\Mitchell_Santner.jpg', 'description': 'Mitchell_Santner'},
            'Shaheen Shah Afridi': {'path': '.\Images\Shaheen_Shah_Afridi.jpg', 'description': 'Shaheen_Shah_Afridi'}
            
        }

    @param.depends('Select_bowlers.value')
    def view(self):
        if not self.Select_bowlers.value:
            return pn.pane.Markdown("Please select 2 of the opener options.")

        images = []
        for animal in self.Select_bowlers.value:
            img = Image.open(self.image_dict[animal]['path'])
            img.thumbnail((200, 200))  # Resize image to a maximum of 200x200
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            images.append(
                pn.Column(
                    pn.pane.HTML(f'''
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}">
                        </div>
                    '''),
                    pn.pane.Markdown(f"**{animal}:** {self.image_dict[animal]['description']}",
                                     css_classes=['image-description']),
                    css_classes=['image-column']
                )
            )

        return pn.Row(*images)
    
Bowlersselector = BowlersImageSelector(Select_bowlers)


# ## Now, we have developed Interactive objects, components and pipeline to get outputs based on selected players

# In[28]:


Select_Opener_idf1 = Openers1.interactive()
Select_Opener_idf = Openers.interactive()
Select_Opener_pipeline = (Select_Opener_idf1[(Select_Opener_idf1.Name.isin(Select_Opener))])
Opener_table = Select_Opener_pipeline.pipe(pn.widgets.Tabulator, pagination='remote', header_filters = False, header_align='center', text_align='center', show_index=False, layout='fit_data')


# In[29]:


Select_Opener_pipeline2 = (
    Select_Opener_idf[(Select_Opener_idf.Name.isin(Select_Opener))]
    .groupby(['Category'])[['Total_Runs','innings_dismissed','Total_balls_faced','Total_boundary_runs']].sum()
    .reset_index()  # Reset the index to make it a DataFrame
    .assign(Combined_Batting_Average=lambda x: round((x['Total_Runs'] / x['innings_dismissed']),0))
    .assign(Combined_Strike_Rate=lambda x: round((x['Total_Runs'] / x['Total_balls_faced']) * 100,0))
    # .assign(Combined_Average_Balls_Faced=lambda x: ROUND((x['Total_balls_faced'] / x['innings_dismissed']) * 100),0)
    .assign(Combined_Boundary_Runs=lambda x: round((x['Total_boundary_runs'] / x['Total_Runs']) * 100,0))

)


# In[30]:


Opener_Combined_Batting_Average = Select_Opener_pipeline2['Combined_Batting_Average'].sum()
Opener_Combined_Strike_Rate = Select_Opener_pipeline2['Combined_Strike_Rate'].sum()
Opener_Combined_Boundary_Runs = Select_Opener_pipeline2['Combined_Boundary_Runs'].sum()


# In[31]:


Opener_Combined_Batting_Averageindi =  pn.indicators.Number(name = "Combined_Batting_Average", value = Opener_Combined_Batting_Average,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

Opener_Combined_Strike_Rateindi =  pn.indicators.Number(name = "Combined_Strike_Rate", value = Opener_Combined_Strike_Rate,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

Opener_Combined_Boundary_Runsindi =  pn.indicators.Number(name = "Combined_Boundary_Percent", value = Opener_Combined_Boundary_Runs, format = '{value:.0f}%',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)


# In[32]:


HAnchors1 = HAnchors[["Name","Batting_Avg","Strike_rate","Bound_percent","Avg_Batting_Pos"]]

# Modify to match HAnchors

Select_HAnchors_idf1 = HAnchors1.interactive()
Select_HAnchors_idf = HAnchors.interactive()
Select_HAnchors_pipeline = (Select_HAnchors_idf1[(Select_HAnchors_idf1.Name.isin(Select_HAnchors))])
HAnchors_table = Select_HAnchors_pipeline.pipe(pn.widgets.Tabulator, pagination='remote', header_filters = False, header_align='center', text_align='center', show_index=False, layout='fit_data')

Select_HAnchors_pipeline2 = (
    Select_HAnchors_idf[(Select_HAnchors_idf.Name.isin(Select_HAnchors))]
    .groupby(['Category'])[['Total_Runs','innings_dismissed','Total_balls_faced','Total_boundary_runs']].sum()
    .reset_index()  # Reset the index to make it a DataFrame
    .assign(Combined_Batting_Average=lambda x: round((x['Total_Runs'] / x['innings_dismissed']),0))
    .assign(Combined_Strike_Rate=lambda x: round((x['Total_Runs'] / x['Total_balls_faced']) * 100,0))
    # .assign(Combined_Average_Balls_Faced=lambda x: ROUND((x['Total_balls_faced'] / x['innings_dismissed']) * 100),0)
    .assign(Combined_Boundary_Runs=lambda x: round((x['Total_boundary_runs'] / x['Total_Runs']) * 100,0))

)

HAnchors_Combined_Batting_Average = Select_HAnchors_pipeline2['Combined_Batting_Average'].sum()
HAnchors_Combined_Strike_Rate = Select_HAnchors_pipeline2['Combined_Strike_Rate'].sum()
HAnchors_Combined_Boundary_Runs = Select_HAnchors_pipeline2['Combined_Boundary_Runs'].sum()

HAnchors_Combined_Batting_Averageindi =  pn.indicators.Number(name = "Combined_Batting_Average", value = HAnchors_Combined_Batting_Average,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

HAnchors_Combined_Strike_Rateindi =  pn.indicators.Number(name = "Combined_Strike_Rate", value = HAnchors_Combined_Strike_Rate,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

HAnchors_Combined_Boundary_Runsindi =  pn.indicators.Number(name = "Combined_Boundary_Percent", value = HAnchors_Combined_Boundary_Runs, format = '{value:.0f}%',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)


# In[33]:


LAnchors1 = LAnchors[["Name","Batting_Avg","Strike_rate","Bound_percent","Avg_Batting_Pos"]]

# Modify to match LAnchors
Select_LAnchors_idf1 = LAnchors1.interactive()
Select_LAnchors_idf = LAnchors.interactive()
Select_LAnchors_pipeline = (Select_LAnchors_idf1[(Select_LAnchors_idf1.Name.isin(Select_LAnchors))])
LAnchors_table = Select_LAnchors_pipeline.pipe(pn.widgets.Tabulator, pagination='remote', header_filters = False, header_align='center', text_align='center', show_index=False, layout='fit_data')

Select_LAnchors_pipeline2 = (
    Select_LAnchors_idf[(Select_LAnchors_idf.Name.isin(Select_LAnchors))]
    .groupby(['Category'])[['Total_Runs','innings_dismissed','Total_balls_faced','Total_boundary_runs']].sum()
    .reset_index()  # Reset the index to make it a DataFrame
    .assign(Combined_Batting_Average=lambda x: round((x['Total_Runs'] / x['innings_dismissed']),0))
    .assign(Combined_Strike_Rate=lambda x: round((x['Total_Runs'] / x['Total_balls_faced']) * 100,0))
    # .assign(Combined_Average_Balls_Faced=lambda x: ROUND((x['Total_balls_faced'] / x['innings_dismissed']) * 100),0)
    .assign(Combined_Boundary_Runs=lambda x: round((x['Total_boundary_runs'] / x['Total_Runs']) * 100,0))

)

LAnchors_Combined_Batting_Average = Select_LAnchors_pipeline2['Combined_Batting_Average'].sum()
LAnchors_Combined_Strike_Rate = Select_LAnchors_pipeline2['Combined_Strike_Rate'].sum()
LAnchors_Combined_Boundary_Runs = Select_LAnchors_pipeline2['Combined_Boundary_Runs'].sum()

LAnchors_Combined_Batting_Averageindi =  pn.indicators.Number(name = "Combined_Batting_Average", value = LAnchors_Combined_Batting_Average,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

LAnchors_Combined_Strike_Rateindi =  pn.indicators.Number(name = "Combined_Strike_Rate", value = LAnchors_Combined_Strike_Rate,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

LAnchors_Combined_Boundary_Runsindi =  pn.indicators.Number(name = "Combined_Boundary_Percent", value = LAnchors_Combined_Boundary_Runs, format = '{value:.0f}%',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)


# In[34]:


Allrounders1 = Allrounders[["Name","Batting_Avg","Strike_rate","Bound_percent","Bowling_Economy","Bowling_strikerate","Bowling_Average","DotballPercent"]]

# Modify to match LAnchors

Select_Allrounders_idf = Allrounders.interactive()
Select_Allrounders_idf1 = Allrounders1.interactive()
Select_Allrounders_pipeline = (Select_Allrounders_idf1[(Select_Allrounders_idf1.Name.isin(Select_Allrounders))])
Allrounders_table = Select_Allrounders_pipeline.pipe(pn.widgets.Tabulator, pagination='remote', header_filters = False, header_align='center', text_align='center', show_index=False, layout='fit_data')

Select_Allrounders_pipeline2 = (
    Select_Allrounders_idf[(Select_Allrounders_idf.Name.isin(Select_Allrounders))]
    .groupby(['Category'])[['Total_Runs','innings_dismissed','Total_balls_faced','Total_runs_conceded','Total_balls','Total_wickets']].sum()
    .reset_index()  # Reset the index to make it a DataFrame
    .assign(Combined_Batting_Average=lambda x: round((x['Total_Runs'] / x['innings_dismissed']),0))
    .assign(Combined_Strike_Rate=lambda x: round((x['Total_Runs'] / x['Total_balls_faced']) * 100,0))
    # .assign(Combined_Average_Balls_Faced=lambda x: ROUND((x['Total_balls_faced'] / x['innings_dismissed']) * 100),0)
    .assign(Combined_Bowling_Economy=lambda x: round((x['Total_runs_conceded'] / (x['Total_balls']/6)),0))
    .assign(Combined_Bowling_Strike_Rate=lambda x: round((x['Total_balls'] / x['Total_wickets']),0))

)

Allrounders_Combined_Batting_Average = Select_Allrounders_pipeline2['Combined_Batting_Average'].sum()
Allrounders_Combined_Strike_Rate = Select_Allrounders_pipeline2['Combined_Strike_Rate'].sum()
Allrounders_Combined_Bowling_Economy = Select_Allrounders_pipeline2['Combined_Bowling_Economy'].sum()
Allrounders_Combined_Bowling_Strike_Rate = Select_Allrounders_pipeline2['Combined_Bowling_Strike_Rate'].sum()

Allrounders_Combined_Batting_Averageindi =  pn.indicators.Number(name = "Combined_Batting_Average", value = Allrounders_Combined_Batting_Average,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

Allrounders_Combined_Strike_Rateindi =  pn.indicators.Number(name = "Combined_Strike_Rate", value = Allrounders_Combined_Strike_Rate,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

Allrounders_Combined_Bowling_Economyindi =  pn.indicators.Number(name = "Combined_Bowling_Economy", value = Allrounders_Combined_Bowling_Economy, 
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

Allrounders_Combined_Bowling_Strike_Rateindi =  pn.indicators.Number(name = "Combined_Bowling_Strike_Rate", value = Allrounders_Combined_Bowling_Strike_Rate,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)


# In[35]:


bowlers1 = bowlers[["Name","Bowling_Economy","Bowling_strikerate","Bowling_Average","DotballPercent"]]

# Modify to match LAnchors

Select_bowlers_idf = bowlers.interactive()
Select_bowlers_idf1 = bowlers1.interactive()
Select_bowlers_pipeline1 = (Select_bowlers_idf1[(Select_bowlers_idf1.Name.isin(Select_bowlers))])
bowlers_table = Select_bowlers_pipeline1.pipe(pn.widgets.Tabulator, pagination='remote', header_filters = False, header_align='center', text_align='center', show_index=False, layout='fit_data')

Select_bowlers_pipeline2 = (
    Select_bowlers_idf[(Select_bowlers_idf.Name.isin(Select_bowlers))]
    .groupby(['Category'])[['Total_runs_conceded','Total_balls','Total_wickets','Ballsnoruns']].sum()
    .reset_index()  # Reset the index to make it a DataFrame
    .assign(Combined_Bowling_Economy=lambda x: round((x['Total_runs_conceded'] / (x['Total_balls']/6)),0))
    .assign(Combined_Bowling_Strike_Rate=lambda x: round((x['Total_balls'] / x['Total_wickets']),0))
    .assign(Combined_Dot_Ball_Percent=lambda x: round((x['Ballsnoruns'] / x['Total_balls'])*100,0))
)

bowlers_Combined_Bowling_Economy = Select_bowlers_pipeline2['Combined_Bowling_Economy'].sum()
bowlers_Combined_Bowling_Strike_Rate = Select_bowlers_pipeline2['Combined_Bowling_Strike_Rate'].sum()
bowlers_Combined_Dot_ball_percent = Select_bowlers_pipeline2['Combined_Dot_Ball_Percent'].sum()

bowlers_Combined_Bowling_Economyindi =  pn.indicators.Number(name = "Combined_Bowling_Economy", value = bowlers_Combined_Bowling_Economy,
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

bowlers_Combined_Bowling_Strike_Rateindi =  pn.indicators.Number(name = "Combined_Bowling_Strike_Rate", value = bowlers_Combined_Bowling_Strike_Rate, 
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)

bowlers_Combined_Dot_ball_percentindi =  pn.indicators.Number(name = "Combined_Doll_Ball_Percent", value = bowlers_Combined_Dot_ball_percent, format = '{value:.0f}%',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                styles = css_sample2)


# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Data Visualization</span>

# In[36]:


pn.param.ParamMethod.loading_indicator = True

GoldenTemplate = pn.template.GoldenTemplate(title = "Cricket Stats Data Dashboard", sidebar_width = 25, main_max_width = "2500px",
                                           header_background = '#5F9EA0', theme=DarkTheme)


# In[37]:


component1 = pn.Column(pn.Row(selector.view),
                       pn.Row(pn.Column(Opener_table, sizing_mode = 'stretch_width')),
                       pn.Row(pn.Column(Opener_Combined_Batting_Averageindi),pn.Spacer(width=50),
                             pn.Column(Opener_Combined_Strike_Rateindi),pn.Spacer(width=50),
                             pn.Column(Opener_Combined_Boundary_Runsindi)),name = "Picking Openers")

component2 = pn.Column(pn.Row(Hanchorselector.view),
                       pn.Row(pn.Column(HAnchors_table, sizing_mode = 'stretch_width')),
                       pn.Row(pn.Column(HAnchors_Combined_Batting_Averageindi),pn.Spacer(width=50),
                             pn.Column(HAnchors_Combined_Strike_Rateindi),pn.Spacer(width=50),
                             pn.Column(HAnchors_Combined_Boundary_Runsindi)),name = "Higher Middle Order")

component3 = pn.Column(pn.Row(LAnchorselector.view),
                       pn.Row(pn.Column(LAnchors_table, sizing_mode = 'stretch_width')),
                       pn.Row(pn.Column(LAnchors_Combined_Batting_Averageindi),pn.Spacer(width=50),
                             pn.Column(LAnchors_Combined_Strike_Rateindi),pn.Spacer(width=50),
                             pn.Column(LAnchors_Combined_Boundary_Runsindi)),name = "Lower Middle Order")

component4 = pn.Column(pn.Row(AllRounderselector.view),
                       pn.Row(pn.Column(Allrounders_table, sizing_mode = 'stretch_width')),
                       pn.Row(pn.Column(Allrounders_Combined_Batting_Averageindi),pn.Spacer(width=50),
                             pn.Column(Allrounders_Combined_Strike_Rateindi),pn.Spacer(width=50),
                             pn.Column(Allrounders_Combined_Bowling_Economyindi),pn.Spacer(width=50),
                             pn.Column(Allrounders_Combined_Bowling_Strike_Rateindi)) ,name = "All Rounders")

component5 = pn.Column(pn.Row(Bowlersselector.view),
                       pn.Row(pn.Column(bowlers_table, sizing_mode = 'stretch_width')),
                       pn.Row(pn.Column(bowlers_Combined_Bowling_Economyindi),pn.Spacer(width=50),
                             pn.Column(bowlers_Combined_Bowling_Strike_Rateindi),pn.Spacer(width=50),
                             pn.Column(bowlers_Combined_Dot_ball_percentindi)),name = "Bowlers")

GoldenTemplate.main.append(component1)
GoldenTemplate.main.append(component2)
GoldenTemplate.main.append(component3)
GoldenTemplate.main.append(component4)
GoldenTemplate.main.append(component5)


# In[38]:


GoldenTemplate.servable()

