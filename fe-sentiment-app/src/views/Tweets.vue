<template>
	<div class="tweets">
		<b-container>
			<h1 class="text-left mt-3">Tweets contain <b class="text-primary">#{{this.hashtag}}</b></h1>
			<div v-if="error" class="error">{{ error }}</div>
			<div v-if="loading" class="loading">Loading...</div>
			<div v-else>
				<b-table striped hover :fields="fields" :items="items" class="mt-4">
					<template v-slot:cell(index)="data">{{ data.index + 1 + ((currentPage-1)*perPage)}}</template>
					<template v-slot:cell(username)="data">
						<b class="text-primary">@{{ data.item.username }}</b>
					</template>
					<template v-slot:cell(tweet)="data">
						<b>{{ data.item.tweet }}</b>
					</template>
					<template v-slot:cell(sentiment)="data">
						<div v-if="data.item.sentiment==='positive'">
							<b class="text-success">{{ data.item.sentiment }}</b>
						</div>
						<div v-else>
							<b class="text-danger">{{ data.item.sentiment }}</b>
						</div>
					</template>
				</b-table>
				<template>
					<div class="mt-3">
						<h6 class="text-center">Pages</h6>
						<b-pagination on-change="changePage" v-model="currentPage" :per-page="perPage" :total-rows="rows" align="center"></b-pagination>
					</div>
				</template>
			</div>
		</b-container>
	</div>
</template>

<script>
	import axios from "axios";
	export default {
		name: "Tweets",
		data: function () {
			return {
				fields: [
					"index",
					{
						key: "username",
						label: "Username"
					},
					{
						key: "tweet",
						label: "Tweet"
					},
					{
						key: "sentiment",
						label: "Sentiment"
					}
				],
				currentPage: 1,
				items: null,
				hashtag : null,
				error : null,
				loading : null,
				perPage : 20,
				rows : null,
			};
		},
		created() {
			this.hashtag = this.$route.params.hashtag;
			this.countTweet();
			this.fetchData(1);
		},
		components: {},
		watch:{
			currentPage : function(page) {
				this.fetchData(page);
			}
		},
		methods: {
			countTweet(){
				let url = "http://127.0.0.1:5000/tweet/count?hashtag="+this.hashtag;
				axios
					.get(url)
					.then(
						function(response){
							this.rows = response.data.count
						}.bind(this)
					)
					.catch(function(error){
						this.error = error;
					});
			},
			fetchData(page){
				let url = "http://127.0.0.1:5000/tweet/all?hashtag="+this.hashtag+"&page="+page;
				axios
					.get(url)
					.then(
						function (response) {
							console.log(response);
							this.loading = false;
							let tableContent = [];
							for (let i = 0; i < response.data.length; i++) {
								let sentiment = "negative"
								if(response.data[i].classification_result==1){
									sentiment = "positive"
								}
								let item = {
									id: response.data[i].id,
									username: response.data[i].user_screen_name,
									tweet: response.data[i].text,
									sentiment: sentiment,
								};
								tableContent.push(item);
							}
							this.items = tableContent; 
						}.bind(this)
					)
					.catch(
						function (error) {
							this.error = error;
						}.bind(this)
					);
			}
		}
	};
</script>