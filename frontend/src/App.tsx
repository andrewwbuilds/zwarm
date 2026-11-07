import { useControls } from './hooks/useControls';
import { useSim } from './hooks/useSim';
import { Console } from './components/Console';
import { Trajectory } from './components/Trajectory';
import { useEffect } from 'react';
import { backendLabel } from './backend';

function App() {
  const controls = useControls();
  const sim = useSim();
  if (!sim.ready) return null;
  return (
    <Console controls={controls} sim={sim}>
      <Trajectory sim={sim.state} />
    </Console>
  );
}

export default App;
