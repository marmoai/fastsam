/**
 * Visual Asset OS - Minimal Viable Decision Runtime (Stage 1)
 * 核心目标：现在只需要记录“上下文”（Action, Target, Prompt），不需要复杂的 Reasoning Tree 或 CTR 分析。
 */

export type ActionType = 
    | 'extract_object' 
    | 'change_style' 
    | 'relocate' 
    | 'remove'
    | 'add_element'
    | 'custom';

export interface DecisionLog {
    logId: string;           
    action: ActionType;      
    targetAssetId?: string;   
    
    // MVR数据：最简单的语境记录
    prompt?: string;         
    context?: string;       
    
    changePayload: Record<string, any>; 
    
    timestamp: number;
}

export class DecisionGraph {
    private logs: DecisionLog[] = [];

    addLog(log: DecisionLog) {
        this.logs.push(log);
    }

    getHistory(): DecisionLog[] {
        return [...this.logs];
    }
}
