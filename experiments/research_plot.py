from pathlib import Path
import json
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG

root = Path(r'D:\CoLaKGResearch')
methods = [('baseline','Original CoLaKG','#222222'),
           ('catalog_contrastive','BPR + catalog CL','#2878B5'),
           ('consensus_residual','Consensus + CF residual','#9467BD'),
           ('semantic_graph_decoder','Separate semantic graph','#E58124'),
           ('aligned_catalog','Aligned catalog decoder','#299D8F'),
           ('pair_geometry','BPR + pair geometry','#C64545')]
rows, data, colors = [], [], []
for method, label, color in methods:
    runs = sorted(root.glob('development-'+method+'-s2020-*'))
    completed = [p for p in runs if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['epochs']==200]
    if not completed:
        continue
    run = completed[-1]
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    evaluations = json.loads((run/'evaluations.json').read_text(encoding='utf-8'))
    data.append([(e['completed_epochs'], e['metrics']['ndcg'][1]) for e in evaluations])
    colors.append(color)
    rows.append({'method':method,'label':label,'run':str(run),
                 'score':result['selection_score_last_three_ndcg20'],
                 'recall20':result['final_metrics']['recall'][1],
                 'final_ndcg20':result['final_metrics']['ndcg'][1],
                 'seconds':result['elapsed_seconds'],
                 'peak_mib':result['peak_allocated_bytes']/2**20,'git_commit':result['git_commit']})

d = Drawing(850,440)
d.add(Rect(0,0,850,440,fillColor=__import__('reportlab').lib.colors.white,strokeColor=None))
d.add(String(54,410,'LastFM: validation only, 200 epochs, seed 2020',fontName='Helvetica-Bold',fontSize=17))
chart=LinePlot();chart.x=62;chart.y=72;chart.width=560;chart.height=295
chart.data=data;chart.joinedLines=1
chart.xValueAxis.valueMin=20;chart.xValueAxis.valueMax=200;chart.xValueAxis.valueSteps=list(range(20,201,20))
low=min(y for series in data for x,y in series);high=max(y for series in data for x,y in series)
chart.yValueAxis.valueMin=int((low-.005)*100)/100;chart.yValueAxis.valueMax=(int((high+.01)*100)+1)/100
chart.yValueAxis.labelTextFormat='%.2f'
chart.xValueAxis.labels.fontSize=10;chart.yValueAxis.labels.fontSize=10
from reportlab.lib.colors import HexColor
for i,color in enumerate(colors):
    chart.lines[i].strokeColor=HexColor(color);chart.lines[i].strokeWidth=2
d.add(chart)
d.add(String(266,39,'Completed training epochs',fontSize=12))
d.add(String(14,380,'NDCG@20',fontSize=11))
for i,row in enumerate(rows):
    y=350-i*46
    d.add(Line(643,y,668,y,strokeColor=HexColor(colors[i]),strokeWidth=2.5))
    d.add(String(675,y-3,row['label'],fontSize=9))
    d.add(String(675,y-18,'Last-three mean: %.5f'%row['score'],fontSize=9))
d.add(String(54,15,'Fit / validation = 46,953 / 10,833; original test labels excluded from development.',fontSize=10))
out=root/'study-v1'/'figures';out.mkdir(exist_ok=True)
renderSVG.drawToFile(d,str(out/'screening-ndcg20.svg'))
svg=out/'screening-ndcg20.svg'
svg.write_text(svg.read_text(encoding='utf-8').replace('font-family: Helvetica-Bold;', 'font-family: Arial; font-weight: bold;').replace('font-family: Helvetica;', 'font-family: Arial;').replace('font-family: Times-Roman;', 'font-family: Arial;'),encoding='utf-8')
(out.parent/'screening-summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))

# A single frozen final test gives no test trajectory; plot training loss only.
final_run=root/'final-pair_geometry-s2020-20261023-171221-e3dfce'
if (final_run/'result.json').exists():
    import math
    final=json.loads((final_run/'result.json').read_text(encoding='utf-8'))
    assert final['success'] and final['completed_epochs']==2000 and final['evaluation_count']==1
    history=json.loads((final_run/'history.json').read_text(encoding='utf-8'))
    loss_plot=Drawing(850,440)
    loss_plot.add(Rect(0,0,850,440,fillColor=__import__('reportlab').lib.colors.white,strokeColor=None))
    loss_plot.add(String(60,410,'CoLaKG + pair geometry: 2000 epochs, seed 2020',fontName='Helvetica-Bold',fontSize=17))
    series=LinePlot();series.x=65;series.y=75;series.width=730;series.height=290
    series.data=[[(r['completed_epochs'],r['loss']) for r in history]];series.joinedLines=1
    series.lines[0].strokeColor=HexColor('#2878B5');series.lines[0].strokeWidth=1
    series.xValueAxis.valueMin=0;series.xValueAxis.valueMax=2000;series.xValueAxis.valueSteps=list(range(0,2001,250))
    series.yValueAxis.valueMin=0;series.yValueAxis.valueMax=math.ceil(max(r['loss'] for r in history)*10)/10
    series.yValueAxis.labelTextFormat='%.1f'
    loss_plot.add(series);loss_plot.add(String(14,380,'Joint loss',fontSize=11));loss_plot.add(String(345,40,'Completed training epochs',fontSize=12))
    loss_plot.add(String(60,16,'BPR + 0.05 * geometry + 1e-4 * L2; absolute loss differs from the original BPR objective.',fontSize=10))
    target=out/'final-training-loss.svg';renderSVG.drawToFile(loss_plot,str(target))
    target.write_text(target.read_text(encoding='utf-8').replace('font-family: Helvetica-Bold;', 'font-family: Arial; font-weight: bold;').replace('font-family: Helvetica;', 'font-family: Arial;').replace('font-family: Times-Roman;', 'font-family: Arial;'),encoding='utf-8')
