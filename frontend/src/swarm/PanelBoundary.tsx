import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  name: string;
  // The grid area the panel normally fills, so a crash message lands in the same place.
  area?: string;
  children: ReactNode;
}

interface State {
  message: string | null;
}

// A crash in one panel stays in that panel. HOLD, LAND and E-STOP sit outside every boundary.
export default class PanelBoundary extends Component<Props, State> {
  state: State = { message: null };

  static getDerivedStateFromError(error: unknown): State {
    return { message: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[zwarm] ${this.props.name} panel crashed`, error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.message === null) return this.props.children;
    return (
      <div className="zw-panel-crash" style={{ gridArea: this.props.area }}>
        <p>The {this.props.name} panel crashed: {this.state.message}</p>
        <p>HOLD, LAND and E-STOP still work.</p>
        <button type="button" className="zw-btn" onClick={() => this.setState({ message: null })}>Reload this panel</button>
      </div>
    );
  }
}
