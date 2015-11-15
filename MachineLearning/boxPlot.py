from bokeh.io import output_notebook, show
from bokeh.charts import BoxPlot
from bokeh.sampledata.autompg import autompg as data


output_notebook()

box = BoxPlot(data, values = 'mpg', label='cyl', title = "MPG Summary (grouped by CYL)")
show(box)