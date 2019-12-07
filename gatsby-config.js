module.exports = {
  siteMetadata: {
    title: `FANG'S NOTEBOOK`,
    description: `About tech, life, and randomness`,
    author: `Fang Cabrera`,
    authorTagline1: `computer science apprentice`,
    authorTagline2: `classical music snob & sci-fi writer wannabe`,
    social: {
      github: `TakaiKinoko`
    }
  },
  plugins: [
    `gatsby-plugin-catch-links`,
    `gatsby-plugin-styled-components`,
    `gatsby-plugin-react-helmet`,
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `images`,
        path: `${__dirname}/src/images`
      }
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `blog`,
        path: `${__dirname}/src/content`
      }
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: `gatsby-remark-images`,
            options: {
              maxWidth: 590,
              showCaptions: true
            }
          },
          {
            resolve: `gatsby-remark-responsive-iframe`,
            options: {
              wrapperStyle: `margin-bottom: 1.0725rem`
            }
          },
          `gatsby-remark-prismjs`,
          `gatsby-remark-copy-linked-files`,
          `gatsby-remark-smartypants`,
          `gatsby-remark-reading-time`
        ]
      }
    },
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    `gatsby-plugin-catch-links`,
    {
      resolve: `gatsby-plugin-google-analytics`,
      options: {
        //trackingId: `ADD YOUR TRACKING ID HERE`,
      }
    },
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: `gatsby-blog-starter`,
        short_name: `starter`,
        start_url: `/`,
        background_color: `#30302F`,
        theme_color: `#30302F`,
        display: `minimal-ui`,
        icon: `src/images/beanie-icon.png` // This path is relative to the root of the site.
      }
    },
    "gatsby-plugin-offline"
  ]
};
