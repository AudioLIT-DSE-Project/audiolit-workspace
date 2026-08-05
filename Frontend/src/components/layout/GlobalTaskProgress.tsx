import React from 'react';
import { useTaskStatus, TaskState } from '../../hooks/useTaskStatus';
// Assuming you are using shadcn-ui for the Progress component
import { Progress } from '../ui/progress'; 

interface GlobalTaskProgressProps {
  taskId: string | null;
  onComplete: () => void;
}

export const GlobalTaskProgress: React.FC<GlobalTaskProgressProps> = ({ taskId, onComplete }) => {
  const { state, error } = useTaskStatus(taskId);

  React.useEffect(() => {
    if (state === 'SUCCESS' || state === 'FAILURE') {
      const timer = setTimeout(() => {
        onComplete();
      }, 1500); 
      return () => clearTimeout(timer);
    }
  }, [state, onComplete]);

  if (!taskId) return null;

  let progressValue = 0;
  let barColor = 'bg-blue-500';
  let statusText = 'Initializing...';

  if (state === 'QUEUED') {
    progressValue = 15;
    statusText = 'Queued for processing...';
  } else if (state === 'PROCESSING') {
    progressValue = 65;
    statusText = 'Running model inference...';
    barColor = 'bg-indigo-500 animate-pulse';
  } else if (state === 'RETRYING') {
    progressValue = 40;
    statusText = 'Encountered error, retrying...';
    barColor = 'bg-yellow-500';
  } else if (state === 'SUCCESS') {
    progressValue = 100;
    statusText = 'Analysis complete!';
    barColor = 'bg-green-500';
  } else if (state === 'FAILURE') {
    progressValue = 100;
    statusText = error || 'Task Failed';
    barColor = 'bg-red-500';
  }

  return (
    <div className="w-full max-w-md mx-auto mt-4 p-4 border rounded-lg shadow-sm bg-white dark:bg-gray-800 transition-all duration-500 ease-in-out">
      <div className="flex justify-between mb-2 text-sm font-medium">
        <span>Task Status</span>
        <span className={state === 'FAILURE' ? 'text-red-500' : 'text-gray-500'}>
          {statusText}
        </span>
      </div>
      <Progress 
        value={progressValue} 
        className={`w-full h-2 ${barColor} transition-all duration-700 ease-out`} 
      />
    </div>
  );
};
