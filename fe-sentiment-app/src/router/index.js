import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../views/Home.vue'
import Hashtag from '../views/Hashtag.vue'
import Tweets from '../views/Tweets.vue'
import Dashboard from '../views/Dashboard.vue'

Vue.use(VueRouter)

  const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/hashtag',
    name: 'Hashtag',
    component: Hashtag
  },
  {
    path: '/tweets/:hashtag?',
    name: 'Tweets',
    component: Tweets
  },
  {
    path: '/dashboard/:hashtag?',
    name: 'Dashboard',
    component: Dashboard
  }
]

const router = new VueRouter({
  routes
})

export default router
