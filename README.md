# Gaze Tracking and Reinforcement Learning Project

## Description
This project implements a gaze tracking system using OpenCV and MediaPipe, combined with reinforcement learning to train an agent for gaze-based actions. The environment simulates gaze detection and provides rewards based on the agent's actions.

## Features
- Real-time gaze tracking using MediaPipe.
- Reinforcement learning with TensorFlow.
- Custom Gym environment for gaze-based actions.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd gaze
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv gaze_env
   .\gaze_env\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Gaze Tracking
Run the gaze tracking script:
```bash
python gaze_tracker.py
```

### Train Agent
Train the reinforcement learning agent:
```bash
python train_agent.py
```

### Test Environment
Test the custom Gym environment:
```bash
python test_env.py
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Feel free to submit issues or pull requests to improve the project.
