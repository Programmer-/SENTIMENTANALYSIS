<template>
	<div class="list">
		<h4>List Hashtag</h4>
		<div v-if="error" class="error">{{ error }}</div>
		<div v-if="loading" class="loading">Loading...</div>
		<div v-else>
			<b-table striped hover :fields="fields" :items="items">
				<template v-slot:cell(index)="data">{{ data.index + 1 }}</template>
				<template v-slot:cell(hashtag)="data">
					<b class="text-info">#{{ data.item.hashtag }}</b>
				</template>
				<template v-slot:cell(action)="data">
					<b-button v-on:click="fetchTweet(data.item.hashtag)" class="mr-3 mb-3" variant="primary">
						Fetch Data
					</b-button>
					<b-button v-on:click="editHashtag(data.item.id, data.item.hashtag)" class="mr-3 mb-3" variant="warning">
						Edit
					</b-button>
					<b-button v-on:click="viewDashboard(data.item.hashtag)" class="mr-3 mb-3" variant="success">
						View Dashboard
					</b-button>
					<b-button v-on:click="viewTweets(data.item.hashtag)" class="mr-3 mb-3" variant="info">
						View Tweets
					</b-button>
					<b-button v-on:click="deleteHashtag(data.item.id)" class="mr-3 mb-3" variant="danger">
						Delete
					</b-button>
				</template>
			</b-table>
		</div>
	</div>
</template>

<script>
	import axios from "axios";
	export default {
		name: "ListHashtag",
		data() {
			return {
				loading: true,
				error: false,
				fields: [
					"index",
					{
						key: "hashtag",
						label: "Hashtag"
					},
					{
						key: "action",
						label: "Action"
					}
				],
				items: null,
				optionAxios : {
					headers: {
						"Access-Control-Allow-Origin": "*",
						"Content-type": "application/json"
					}
				},
			};
		},
		created() {
			this.fetchData();
		},
		watch: {
			$route: "fetchData"
		},
		methods: {
			fetchTweet: function (hashtag) {
				this.error = null;
				this.loading = true;
				let url = "http://127.0.0.1:5000/tweet/crawl";
				axios.post(url,{
					hashtag : hashtag
				},this.optionAxios).then(function(response){
					console.log(response);
					this.$router.push({path:`/tweets/${hashtag}`});
				}.bind(this)
				).catch(function(error){
					this.error = error;
				}.bind(this));

			},
			editHashtag: function (id, hashtag) {
				this.$emit("actionButton", id, hashtag);
			},
			viewDashboard: function (hashtag) {
				this.$router.push({path:`/dashboard/${hashtag}`});
			},
			viewTweets: function(hashtag){
				console.log(hashtag);
				this.$router.push({path:`/tweets/${hashtag}`});
			},
			deleteHashtag: function (id) {
				this.error = null;
				this.loading = true;
				let url = "http://127.0.0.1:5000/hashtag/delete";
				axios.post(url,{
					id : id
				},this.optionAxios).then(function(response){
					console.log(response);
					this.loading = false;
					this.fetchData();	
				}.bind(this)
				).catch(function(error){
					console.log(error)
					this.error = true;
				}.bind(this));
			},
			fetchData() {
				this.error = null;
				this.loading = true;
				let url = "http://127.0.0.1:5000/hashtag/all";
				axios
					.get(url)
					.then(
						function (response) {
							console.log(response);
							this.loading = false;
							let tableContent = [];
							for (let i = 0; i < response.data.length; i++) {
								let item = {
									id: response.data[i].id,
									hashtag: response.data[i].hashtag
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