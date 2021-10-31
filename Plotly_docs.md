## Plotly 는?
* 파이썬의 대표적인 인터랙티브 시각화 도구
* [Plotly Python Graphing Library | Python | Plotly](https://plotly.com/python/)
* [Financial Charts | Python | Plotly](https://plotly.com/python/financial-charts/)
* [Time Series and Date Axes | Python | Plotly](https://plotly.com/python/time-series/)
* [OHLC Charts | Python | Plotly](https://plotly.com/python/ohlc-charts/)
* [Python API reference for plotly — 4.14.3 documentation](https://plotly.com/python-api-reference/)

###  Plotly Express: high-level interface for data visualization
* https://plotly.com/python-api-reference/plotly.express.html
* seaborn 과 비슷한 사용법
* 사용법이 plotly.graph_objects 에 비해 비교적 간단한 편 

#### 사용법

* **scatter([data_frame, x, y, color, symbol, …])**
* scatter_3d([data_frame, x, y, z, color, …])
* scatter_polar([data_frame, r, theta, color, …])
* scatter_ternary([data_frame, a, b, c, …])
* scatter_mapbox([data_frame, lat, lon, …])
* scatter_geo([data_frame, lat, lon, …])
* **line([data_frame, x, y, line_group, color, …])**
* line_3d([data_frame, x, y, z, color, …])
* line_polar([data_frame, r, theta, color, …])
* line_ternary([data_frame, a, b, c, color, …])
* line_mapbox([data_frame, lat, lon, color, …])
* line_geo([data_frame, lat, lon, locations, …])
* **area([data_frame, x, y, line_group, color, …])**
* **bar([data_frame, x, y, color, facet_row, …])**
* timeline([data_frame, x_start, x_end, y, …])
* bar_polar([data_frame, r, theta, color, …])
* **violin([data_frame, x, y, color, facet_row, …])**
* **box([data_frame, x, y, color, facet_row, …])**
* **strip([data_frame, x, y, color, facet_row, …])**
* **histogram([data_frame, x, y, color, …])**
* **pie([data_frame, names, values, color, …])**
* **treemap([data_frame, names, values, …])**
* **sunburst([data_frame, names, values, …])**
* funnel([data_frame, x, y, color, facet_row, …])
* funnel_area([data_frame, names, values, …])
* scatter_matrix([data_frame, dimensions, …])
* parallel_coordinates([data_frame, …])
* parallel_categories([data_frame, …])
* choropleth([data_frame, lat, lon, …])
* choropleth_mapbox([data_frame, geojson, …])
* density_contour([data_frame, x, y, z, …])
* density_heatmap([data_frame, x, y, z, …])
* density_mapbox([data_frame, lat, lon, z, …])
* imshow(img[, zmin, zmax, origin, labels, x, …])

### Graph Objects: low-level interface to figures, traces and layout
* https://plotly.com/python-api-reference/plotly.graph_objects.html
* 다양한 시각화

#### Figure
Figure([data, layout, frames, skip_invalid])

#### Layout
Layout([arg, activeshape, angularaxis, …])

#### Simple Traces
* Scatter([arg, cliponaxis, connectgaps, …])
* Scattergl([arg, connectgaps, customdata, …])
* Bar([arg, alignmentgroup, base, basesrc, …])
* Pie([arg, automargin, customdata, …])
* Heatmap([arg, autocolorscale, coloraxis, …])
* Heatmapgl([arg, autocolorscale, coloraxis, …])
* Image([arg, colormodel, customdata, …])
* Contour([arg, autocolorscale, autocontour, …])
* Table([arg, cells, columnorder, …])

#### Distribution Traces
* Box([arg, alignmentgroup, boxmean, …])
* Violin([arg, alignmentgroup, bandwidth, …])
* Histogram([arg, alignmentgroup, autobinx, …])
* Histogram2d([arg, autobinx, autobiny, …])
* Histogram2dContour([arg, autobinx, …])

#### Finance Traces
* Ohlc([arg, close, closesrc, customdata, …])
* Candlestick([arg, close, closesrc, …])
* Waterfall([arg, alignmentgroup, base, …])
* Funnel([arg, alignmentgroup, cliponaxis, …])
* Funnelarea([arg, aspectratio, baseratio, …])
* Indicator([arg, align, customdata, …])

#### 3D Traces –

* Scatter3d([arg, connectgaps, customdata, …])
* Surface([arg, autocolorscale, cauto, cmax, …])
* Mesh3d([arg, alphahull, autocolorscale, …])
* Cone([arg, anchor, autocolorscale, cauto, …])
* Streamtube([arg, autocolorscale, cauto, …])
* Volume([arg, autocolorscale, caps, cauto, …])
* Isosurface([arg, autocolorscale, caps, …])

#### Map Traces
* Scattergeo([arg, connectgaps, customdata, …])
* Choropleth([arg, autocolorscale, coloraxis, …])
* Scattermapbox([arg, below, connectgaps, …])
* Choroplethmapbox([arg, autocolorscale, …])
* Densitymapbox([arg, autocolorscale, below, …])

#### Specialized Traces
* Scatterpolar([arg, cliponaxis, connectgaps, …])
* Scatterpolargl([arg, connectgaps, …])
* Barpolar([arg, base, basesrc, customdata, …])
* Scatterternary([arg, a, asrc, b, bsrc, c, …])
* Sunburst([arg, branchvalues, count, …])
* Treemap([arg, branchvalues, count, …])
* Sankey([arg, arrangement, customdata, …])
* Splom([arg, customdata, customdatasrc, …])
* Parcats([arg, arrangement, bundlecolors, …])
* Parcoords([arg, customdata, customdatasrc, …])
* Carpet([arg, a, a0, aaxis, asrc, b, b0, …])
* Scattercarpet([arg, a, asrc, b, bsrc, …])
* Contourcarpet([arg, a, a0, asrc, atype, …])

### Pandas와 쉽게 호환되는 cufflinks
* [santosjorge/cufflinks: Productivity Tools for Plotly + Pandas](https://github.com/santosjorge/cufflinks)



## plotly 예제 따라하기
* [Time Series and Date Axes | Python | Plotly](https://plotly.com/python/time-series/)



##  Range Slider와 함께 시계열 그래프 그리기
* [Time Series and Date Axes | Python | Plotly](https://plotly.com/python/time-series/)



### 캔들차트

* [Candlestick Charts | Python | Plotly](https://plotly.com/python/candlestick-charts/)

## OHLC(Open-High-Low-Close)


* [OHLC Charts | Python | Plotly](https://plotly.com/python/ohlc-charts/)

## Candlestick without Rangeslider
* [candlestick Traces | Python | Plotly](https://plotly.com/python/reference/candlestick/)

###  Range Slider와 함께 Candlestick 그리기
* [Time Series and Date Axes | Python | Plotly](https://plotly.com/python/time-series/)





## Plotly 공식문서 더 보기

[Plotly Python Graphing Library | Python | Plotly](https://plotly.com/python/)