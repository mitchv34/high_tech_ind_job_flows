# O*Net Data

I want to create crosswalks between O*Net and other data sources ACS OCC and SOC
1. O*Net => SOC2000 => OCC2000
2. O*Net => SOC2010 => OCC2010
3. O*Net => SOC2018 => OCC2018

Then I want to create scores for each occupation based on the O*Net data. I will use the O*Net data to create a score for each occupation based on the following criteria:

- Knowledge
- Skills (Not done yet)

I've found that not all occupations have a score for each of the knowledge areas. For those occupations that do not have a score for a particular knowledge area, I will use the average score for that knowledge across related occupations. O*Net provides a list of related occupations for each occupation. I will use the related occupations to create the average score for each knowledge area.