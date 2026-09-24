# Football Performance Analyst

The software analyzes 6v6 football matches and generates performance
statistics for each team, such as ball possession, positioning and
overall performance.

> Status: Early stage of development

## Technologies
- Python
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) – object detection
- [Roboflow](https://roboflow.com) – dataset management and annotation

## How it works
1. Input: match video recording (e.g. `.mp4`)
2. Detection of players and the ball on each frame (YOLOv8)
3. Assigning players to teams
4. Calculating statistics
5. Output: team statistics (and optionally an annotated video)

## Dataset
- Annotated with Roboflow
- Planned classes: player, ball, goalkeeper, referee

## Installation
```bash
git clone https://github.com/hubixek/Football-performance-analyst.git
cd Football-performance-analyst
```
Full installation instructions will be added as the project develops.

## Usage
Coming soon.

## TODO
- [ ] Player and ball detection
- [ ] Team assignment (e.g. by jersey color)
- [ ] Ball possession stats
- [ ] Player positioning
- [ ] Overall team performance score
- [ ] Export results to CSV
- [ ] Add `requirements.txt` and usage instructions

## Author
Hubert – [GitHub](https://github.com/hubixek)
