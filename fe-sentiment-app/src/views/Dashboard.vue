<template>
	<div class="dashboard">
		<b-container>
			<h1 class="mt-5 text-left">Dashboard for #{{hashtag}}</h1>
			<hr/>
			<b-row class="mt-1 mb-5">
				<b-col cols="4">
					<b-card title="Number Of Tweets">
						<b-card-text>
							<h1>{{count/1000}}k</h1>
						</b-card-text>
					</b-card>
				</b-col>
				<b-col cols="4">
					<b-card title="Positive Sentiments">
						<b-card-text>
							<h1>{{positiveSentiment}}</h1>
						</b-card-text>
					</b-card>
				</b-col>
				<b-col cols="4">
					<b-card title="Negative Sentiments">
						<b-card-text>
							<h1>{{negativeSentiment}}</h1>
						</b-card-text>
					</b-card>
				</b-col>
			</b-row>
			<div class="mb-5">
				<line-chart v-if="hashtag" :hashtag="hashtag" />
			</div>
			<b-row class="mt-5">
				<b-col cols="5">
					<pie-chart v-if="posneg" :data="posneg"/>
				</b-col>
				<b-col cols="7">
					<top-tweets v-if="hashtag" :hashtag="hashtag" />
				</b-col>
			</b-row>

		</b-container>
	</div>
</template>

<script>
	import LineChart from '@/components/LineChart.vue';
	import PieChart from '@/components/PieChart.vue';
	import TopTweets from '@/components/TopTweets.vue';
	import axios from "axios";
	export default {
		name: 'Dashboard',
		components: {
			'line-chart': LineChart,
			'pie-chart': PieChart,
			'top-tweets': TopTweets,
		},
		data() {
			return {
				count: null,
				positiveSentiment: null,
				negativeSentiment: null,
				hashtag: null,
				posneg : null,
			}
		},
		created(){
			this.hashtag = this.$route.params.hashtag;
			this.countTweet();
			this.countSentiment();
		},
		methods: {
			countTweet: function () {
				let url = "http://127.0.0.1:5000/tweet/count?hashtag=" + this.hashtag;
				axios
					.get(url)
					.then(
						function (response) {
							console.log(response);
							this.count = response.data.count
						}.bind(this)
					)
					.catch(function (error) {
						console.log(error)
					});
			},
			countSentiment: function(){
				let url = "http://127.0.0.1:5000/tweet/sentiment?hashtag=" + this.hashtag;
				axios
					.get(url)
					.then(
						function (response) {
							console.log("a", response);
							for(let i=0; i<response.data.result.length;i++){
								if(response.data.result[i].label==0){
									this.negativeSentiment = response.data.result[i].result;
								}else{
									this.positiveSentiment = response.data.result[i].result;
								}
							}
							this.posneg = [this.positiveSentiment, this.negativeSentiment];
						}.bind(this)
					)
					.catch(function (error) {
						console.log(error)
					});
			}
		}
	}
</script>