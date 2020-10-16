<template>
  <div class="hashtag">
    <b-container>
      <h1 class="text-left mt-3">Hashtag</h1>
      <hr />
      <b-row>
        <b-col cols="2">
          <SubmenuHashtag v-on:childToParent="onChildClick" />
        </b-col>
        <b-col cols="10">
          <div v-if="menu === 2">
            <ListHashtag v-on:actionButton="actionButton" />
          </div>
          <div v-else-if="menu === 1">
            <FormNewHashtag :hashtag="hashtag" v-on:childToParent="onChildClick" />
          </div>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import SubmenuHashtag from "@/components/SubmenuHashtag.vue";
import ListHashtag from "@/components/ListHashtag.vue";
import FormNewHashtag from "@/components/FormNewHashtag.vue";

export default {
  name: "Hashtag",
  data: function() {
    return {
      menu: 2,
      hashtag: null,
      id: null
    };
  },
  components: {
    SubmenuHashtag: SubmenuHashtag,
    ListHashtag: ListHashtag,
    FormNewHashtag: FormNewHashtag
  },
  methods: {
    onChildClick(value) {
      this.menu = value;
      this.hashtag = null;
    },
    actionButton(id, hashtag) {
      this.id = id;
      this.hashtag = hashtag;
      this.menu = 1;
    }
  }
};
</script>