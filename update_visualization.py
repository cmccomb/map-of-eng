import matplotlib.colors  # for getting pretty colors
import matplotlib.pyplot  # for converting rgb to hex
import numpy  # for generic operations
import pandas  # dumping json into csv
import plotly.express  # plotly


# Load the data
all_the_data = pandas.read_csv("data.csv")
x_data = all_the_data["x"].values
y_data = all_the_data["y"].values

all_the_data.sort_values(["department", "faculty"], inplace=True)

# Plot the embeddings
fig = plotly.express.scatter(
    all_the_data,
    x="x", y="y", hover_data=["title", "faculty"],
    color="department",
    symbol_sequence=["circle"]
)

# Make sure the axes are appropriately scaled
fig.update_xaxes(visible=False, autorange=False, range=[numpy.min(x_data) * 1.05, numpy.max(x_data) * 1.05])

fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, range=[numpy.min(y_data) * 1.05, numpy.max(y_data) * 1.05])

# Reset the layout
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.5, itemsizing='constant'), plot_bgcolor="#191C1F", )
fig.update_traces(marker=dict(size=2))
# Remove the logo
fig.show(config=dict(displaylogo=False))

# Save the file
fig.write_html("index.html")
