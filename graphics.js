google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawLineColors);

function drawLineColors() {
  var data = new google.visualization.DataTable();
  data.addColumn('number', 'X');
  data.addColumn('number', 'Iterable');
  data.addColumn('number', 'Parallel');
  data.addColumn('number', 'Tiling');
  data.addColumn('number', 'Tiling 3d');

  data.addRows([
    [1,17.9972,8.40578,9.6754,7.99101],
    [2,18.9222,7.60442,9.2414,7.51112],
    [3,18.0002,8.78542,9.4454,7.79101],
    [4,17.5642,8.31565,9.3733,8.00001],
    [5,16.8892,8.46433,9.5554,7.89121]
  ]);

  var options = {
    hAxis: {
      title: 'Try'
    },
    vAxis: {
      title: 'Time',
      hAxis: { 
        gridlines: {
          color: '#333',
          count: 4
        }
      }
    }
  };

  var chart = new google.visualization.ColumnChart(document.getElementById('chart_div'));
  chart.draw(data, options);
}