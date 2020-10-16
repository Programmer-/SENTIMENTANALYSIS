<template>
    <div>
        <b-form inline>
            <label class="sr-only" for="inline-form-input-name">Hashtag</label>
            <b-input v-model="hashtagValue" id="inline-form-input-name" class="mb-2 mr-sm-2 mb-sm-0" placeholder="#Hashtag"></b-input>
            <b-button v-on:click="save" variant="primary">Save</b-button>
        </b-form>
    </div>
</template>

<script>
import axios from 'axios'
export default {
    name : 'FormNewHashtag',
    props : {
        hashtag : String
    },
    data() {
        return{
            hashtagValue: this.hashtag
        }
    },
    methods :{
        save : function(){
            let optionAxios = {
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Content-type': 'application/json'
                }
            }
            axios.post('http://127.0.0.1:5000/hashtag/insert', {
                hashtag: this.hashtagValue,
            },optionAxios)
            .then(function (response) {
                console.log(response);
                this.$emit('childToParent', 2)
            }.bind(this))
            .catch(function (error) {
                console.log(error);
            });
        }
    }
}
</script>