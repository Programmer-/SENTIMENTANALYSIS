<template>
    <div>
        <b-navbar toggleable="lg" type="dark" variant="primary">
            <b-navbar-brand href="#">Sentiment App</b-navbar-brand>

            <b-navbar-toggle target="nav-collapse"></b-navbar-toggle>

            <b-collapse id="nav-collapse" is-nav>
                <b-navbar-nav>
                    <b-nav-item to="/">Home</b-nav-item>
                    <b-nav-item to="/hashtag">Hashtag</b-nav-item>
                </b-navbar-nav>
                
            </b-collapse>
        </b-navbar>
    </div>
</template>

<script>

export default {
  name: 'NavBar'
}
</script>