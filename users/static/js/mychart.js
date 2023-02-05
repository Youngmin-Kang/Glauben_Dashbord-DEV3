import "./styles.css";
import { Chart } from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";
import "chartjs-plugin-labels";
import * as Gauge from "chartjs-gauge";
import { useEffect } from "react";


ctx = document
    .getElementById('myChart')
    .getContext('2d');
datos = document
    .getElementById('myChartLine')
    .getContext('2d');
ctx1 = document
    .getElementById("myChartSDI")
    .getContext("2d");

const myChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
        labels: [
            'SDI entrada RO', 'Flujo normalizado'
        ],
        datasets: [
            {
                label: 'Datos analizados',
                data: [
                    3, 15
                ],
                backgroundColor: [
                    'rgba(255, 99, 132, 1)', 'rgba(255, 159, 64, 1)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)', 'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        interaction: {
            mode: 'index',
            intersect: false
        }
    }
});

var myChart1 = new Chart(datos, {
    type: 'bar',
    data: {
        labels: [
            'SDI entrada RO', 'Flujo normalizado'
        ],
        datasets: [
            {
                label: 'Datos analizados',
                data: [
                    3, 15
                ],
                backgroundColor: [
                    'rgba(255, 99, 132, 1)', 'rgba(255, 159, 64, 1)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)', 'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true
    }
});

var data = [40, 70, 100];
var value = 32;

var config = new Chart(ctx1,{
  type: "tsgauge",
  data: {
    labels: ["Normal", "Warning", "Critical"],
    datasets: [
      {
        label: "Current Appeal Risk",
        data: data,
        value: value,
        minValue: 0,
        backgroundColor: ["green", "orange", "red"],
        borderWidth: 2
      }
    ]
  },
  options: {
    legend: {
      display: true,
      position: "bottom",
      labels: {
        generateLabels: {}
      }
    },
    responsive: true,
    title: {
      display: true,
      text: "Financial Risk"
    },
    layout: {
      padding: {
        bottom: 20
      }
    },
    needle: {
      radiusPercentage: 1,
      widthPercentage: 1,
      lengthPercentage: 60,
      color: "rgba(0, 0, 0, 1)"
    },
    valueLabel: {
      fontSize: 12,
      formatter: function (value, context) {
        // debugger;
        return value + "X";
        // return '< ' + Math.round(value);
      }
    },
    plugins: {
      datalabels: {
        display: "auto",
        formatter: function (value, context) {
          // debugger;
          return context.chart.data.labels[context.dataIndex];
          // return context.dataIndex===0?'Normal':context.dataIndex===1?'Warning':'Critical';
          // return '< ' + Math.round(value);
        },
        color: function (context) {
          return "white";
        },
        //color: 'rgba(255, 255, 255, 1.0)',
        // backgroundColor: 'rgba(0, 0, 0, 1.0)',
        // borderWidth: 0,
        // borderRadius: 5,
        font: function (context) {
          var innerRadius = Math.round(context.chart.innerRadius);
          console.log(innerRadius);
          var size = Math.round(innerRadius / 8);

          return {
            weight: "normal",
            size: size
          };
        }
        // font: {
        //   weight: 'normal',
        //   size:16
        // }
      }
    }
  }
});
