<template>
    <div class="submenu">
        <b-list-group>
            <b-list-group-item href="#" v-on:click="menu(1)" >New Hashtag</b-list-group-item>
            <b-list-group-item href="#" v-on:click="menu(2)">List Hashtag</b-list-group-item>
        </b-list-group>
    </div>
</template>

<script>
export default {
  name: 'SubmenuHashtag',
  methods: {
    menu: function (index) {
      this.$emit('childToParent', index)
    }
  },
}
</script>

