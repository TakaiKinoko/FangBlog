import React from "react";
import styled from "styled-components";
import { StaticQuery, graphql } from "gatsby";

import media from "../utils/media";
import Github from "../images/social/github.svg";
import Hat from "../images/social/hat.svg";
import LinkedIn from "../images/social/linkedin.svg";
//import Beanie from "../images/beanie-icon.png";
import Nichijou from "../images/Nichijou.svg";

const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 5rem 0;

  ${media.tablet`
    flex-direction: column;
    text-align: center;
  `}
`;

const TextContainer = styled.div`
  ${media.phone`
    order: 2;
  `}
`;

const ImageContainer = styled.div`
  ${media.phone`
    order: 1;
  `}
`;

const Name = styled.h3`
  color: teal;
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: 0.1rem;
  text-transform: uppercase;
  margin: 0;

  ${media.phone`
    text-align: center;
  `}
`;

const TagLine = styled.sub`
  color: black;
  font-weight: normal;
  font-size: 1.6rem;
  margin: 0;
  display: block;
`;

const FangCabIcon = styled.img`
  height: 1.5rem;
  width: 1.5rem;
  padding: 1.5rem 1rem;
`;

const GithubIcon = styled.img`
  height: 1.5rem;
  width: 1.5rem;
  padding: 1.5rem 1rem;
`;

const LinkedInIcon = styled.img`
  height: 1.5rem;
  width: 1.5rem;
  padding: 1.5rem 1rem;
`;

/*
const BeanieIcon = styled.img`
  height: 16rem;
  width: 16rem;
  padding: 1.5rem 1rem;
`; */

const NichijouIcon = styled.img`
  height: 20rem;
  width: 20rem;
  padding: 1.5rem 1rem;
`;

const Bio = () => (
  <StaticQuery
    query={bioQuery}
    render={data => {
      const {
        author,
        authorTagline1,
        authorTagline2,
        social
      } = data.site.siteMetadata;
      return (
        <Container>
          <TextContainer>
            <Name>{author}</Name>
            <TagLine>{authorTagline1}</TagLine>
            <TagLine>{authorTagline2}</TagLine>
            <a
              href={`https://github.com/${social.github}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <GithubIcon src={Github} alt="github" />
            </a>
            <a
              href={`https://www.linkedin.com/in/fang-cabrera-69177587/`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <LinkedInIcon src={LinkedIn} alt="linkedin" />
            </a>
            <a
              href={`https://fangcabrera.com`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <FangCabIcon src={Hat} alt="fangWebsite" />
            </a>
          </TextContainer>
          <ImageContainer>
            <NichijouIcon src={Nichijou} alt={author} />
          </ImageContainer>
        </Container>
      );
    }}
  />
);

const bioQuery = graphql`
  query BioQuery {
    avatar: file(absolutePath: { regex: "/beanie-icon.png/" }) {
      childImageSharp {
        fixed(width: 70, height: 70) {
          ...GatsbyImageSharpFixed
        }
      }
    }
    site {
      siteMetadata {
        author
        authorTagline1
        authorTagline2
        social {
          github
        }
      }
    }
  }
`;

export default Bio;
