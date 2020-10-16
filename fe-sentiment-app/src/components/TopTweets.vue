<template>
	<div class="list">
		<h4>Top Tweets</h4>
		<div v-if="loading">
			Loading...
		</div>
		<div v-else>
			<b-table striped hover :items="items"></b-table>
		</div>
	</div>
</template>

<script>
	import axios from 'axios';
	export default {
		name: 'TopTweets',
		data() {
			return {
				items: null,
				loading: false
			}
		},
		props : {
			hashtag : String
		},
		created(){
			this.fetchData();
		},
		methods: {
			fetchData: function(){
				this.loading = true;
				let url = "http://127.0.0.1:5000/tweet/toptweet?hashtag=" + this.hashtag;
				axios
					.get(url)
					.then(
						function (response) {
							this.loading = false;
							let tableContent = [];
							console.log(response);
							for(let i=0; i<response.data.result.length; i++)
							{
								let row = {
									"username" : response.data.result[i].user_screen_name,
									"tweet" : response.data.result[i].text,
									"retweet" : response.data.result[i].retweet_count
								}
								tableContent.push(row);
							}
							this.items = tableContent;
						}.bind(this)
					)
					.catch(function (error) {
						this.loading = false;
						console.log(error);
					}.bind(this));
			}
		}
	}
</script>