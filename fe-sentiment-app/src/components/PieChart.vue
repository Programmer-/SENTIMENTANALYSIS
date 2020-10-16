<script>
	import {
		Doughnut
	} from "vue-chartjs";
	export default {
		extends: Doughnut,
		props: {
			data: Array
		},
		data() {
			return {
				chartOptions: {
					hoverBorderWidth: 20
				},
				chartData: {
					hoverBackgroundColor: "red",
					hoverBorderWidth: 10,
					labels: ["Green", "Red"],
					datasets: [{
						label: "Data One",
						backgroundColor: ["#41B883", "#E46651"],
						data: [10, 5]
					}]
				}
			};
		},
		mounted() {
			//renderChart function renders the chart with the datacollection and options object.
			this.renderChart(this.chartData, this.chartOptions);
		},
		created(){
			console.log(this.data);
			this.chartData = {
				hoverBackgroundColor: "red",
				hoverBorderWidth: 10,
				labels: ["Positive", "Negative"],
				datasets: [{
					label: "Sentiment",
					backgroundColor: ["#41B883", "#E46651"],
					data: this.data
				}]
			}
		}
	};
</script>