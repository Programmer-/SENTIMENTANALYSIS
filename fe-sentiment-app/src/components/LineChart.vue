<script>
	import {Line} from "vue-chartjs";
	import axios from 'axios';
	export default {
		extends: Line,
		props: {
			hashtag: String
		},
		data() {
			return {
				datacollection: null,
				//Chart.js options that controls the appearance of the chart
				options: {
					scales: {
						yAxes: [{
							ticks: {
								beginAtZero: true
							},
							gridLines: {
								display: true
							}
						}],
						xAxes: [{
							gridLines: {
								display: false
							}
						}]
					},
					legend: {
						display: true
					},
					responsive: true,
					maintainAspectRatio: false
				}
			};
		},
		mounted() {
			
		},
		created() {
			this.fetchData();
		},
		watch: {
			datacollection(){
				this.renderChart(this.datacollection, this.options);
			}
		},
		methods: {
			fetchData: function(){
				let url = "http://127.0.0.1:5000/tweet/daytoday?hashtag=" + this.hashtag;
				axios
					.get(url)
					.then(
						function (response) {
							let label = [];
							let dataPositive = [];
							let dataNegative = [];
							for(let i=0;i<response.data.result.length;i++)
							{
								let result = response.data.result[i];
								if(!label.includes(result.dateLabel)){
									label.push(result.dateLabel);
								}
								if(result.classification_result==0){
									dataNegative.push(parseInt(result.countSentiment));
								}else{
									dataPositive.push(parseInt(result.countSentiment));
								}
							}
							let datacollection = {
								labels: label,
								datasets: [{
										label: "Positive Sentiment",
										backgroundColor: "#27ae60",
										borderColor: "#27ae60",
										fill: false,
										data: dataPositive
									},
									{
										label: "Negative Sentiment",
										backgroundColor: "#f87979",
										borderColor: "#f87979",
										fill: false,
										data: dataNegative
									}
								]
							}
							this.datacollection = datacollection;
							console.log(this.datacollection);
						}.bind(this)
					)
					.catch(function (error) {
						console.log(error)
					});
			}
		}
	};
</script>