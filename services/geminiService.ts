// FIX: We removed the direct Google SDK import.
// The Frontend now acts as a dumb terminal; the Backend holds the Intelligence.

const KERNEL_URL = 'http://localhost:8000';

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export const sendMessageToJarvis = async (
  message: string, 
  history: ChatMessage[],
  contextData: any
): Promise<string> => {
  try {
    // 1. Pack the Context
    const contextSummary = {
      view: contextData.view,
      activeAlerts: contextData.activeAlerts?.length || 0,
      activeModels: contextData.activeModels?.length || 0,
      joinedSKUs: contextData.dataStats?.joinedSKUs || 0
    };

    // 2. Call the Kernel (Backend)
    const response = await fetch(`${KERNEL_URL}/intelligence/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        context: contextSummary
      }),
    });

    if (!response.ok) {
        throw new Error("Kernel refused connection.");
    }

    const data = await response.json();
    return data.response || "No data received from Neural Net.";

  } catch (error) {
    console.error("Jarvis Error:", error);
    return "Intelligence Plane unreachable. Please check Kernel connection.";
  }
};
