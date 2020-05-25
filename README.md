# DL_torch
 
http://tangshusen.me/Dive-into-DL-PyTorch/#/

# pandas show tricks

#Using hide_index() from the style function
df.head(10).style.hide_index()

#Using hide_columns to hide the unnecesary columns
df.head(10).style.hide_index().hide_columns(['method','year'])

#Highlight the maximum number for each column
df.head(10).style.highlight_max(color = 'yellow')
df.head(10).style.highlight_min(color = 'lightblue')

#Adding Axis = 1 to change the direction from column to row
planets.head(10).style.highlight_max(color = 'yellow', axis =1)

#Higlight the null value
planets.head(10).style.highlight_null(null_color = 'red')
