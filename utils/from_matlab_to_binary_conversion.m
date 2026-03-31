%[text] ## **Matlab to binary converter**
%[text] This script converts high-density, single-channel `.mat` recordings (e.g., 32 kHz microelectrode data) into a unified, multiplexed binary file (`.bin`). The output format is explicitly designed for seamless integration with Python's **SpikeInterface** via the `read_binary` function.
%[text] **Core Functionalities**
%[text] - **Memory Efficiency**: Processes high-frequency data in chunks to prevent RAM overflow.
%[text] - **Automated Mirroring**: Dynamically detects and replicates the input folder hierarchy (`Subject/Session/Wideband`) into the output directory.
%[text] - **Native Multiplexing**: Leverages MATLAB's column-major `fwrite` to output interleaved binaries, ensuring direct compatibility with SpikeInterface.
%[text] - **Safe Execution**: Automatically skips existing `.bin` files to prevent unintended overwrites. \
%[text] **Input Data Organization**To ensure correct execution, the input directory must strictly follow this structure:
%[text] 1. **Root**: `raw_matlab/`
%[text] 2. **Subject Level**: Folders named by subject (e.g., `Router/`)
%[text] 3. **Session Level**: Folders named by session (e.g., `Router_20220211/`)
%[text] 4. **Data Level**: A `Wideband/` subfolder containing 1D array files named exactly `[SessionName]_[ChannelNumber].mat`. \
%[text] ### Configuration parameters
% Base paths configuration
base_dir = "C:\Users\tommy\OneDrive - Scuola Superiore Sant'Anna\Monkeys Parma";
matlab_input_dir = fullfile(base_dir, 'raw_matlab');
binary_output_dir = fullfile(base_dir, 'raw_binary');

% Processing parameters
num_channels = 128;
dtype = 'single'; 
chunk_size = 1e7; 
%%
%[text] ### Signals conversion
% 1. Inspect subject directories
subjects = dir(matlab_input_dir);
subjects = subjects([subjects.isdir]);
subjects = subjects(~ismember({subjects.name}, {'.', '..'}));

for s = 1:length(subjects) %[output:group:6e441575]
    subject_name = subjects(s).name;
    subject_in_dir = fullfile(matlab_input_dir, subject_name);
    subject_out_dir = fullfile(binary_output_dir, subject_name);
    
    % 2. Inspect session directories within each subject
    sessions = dir(subject_in_dir);
    sessions = sessions([sessions.isdir]);
    sessions = sessions(~ismember({sessions.name}, {'.', '..'}));
    
    for sess = 1:length(sessions)
        session_name = sessions(sess).name;
        
        in_session_dir = fullfile(subject_in_dir, session_name);
        out_session_dir = fullfile(subject_out_dir, session_name);
        
        in_wideband_dir = fullfile(in_session_dir, 'Wideband');
        in_events_dir = fullfile(in_session_dir, 'Events');
        
        out_wideband_dir = fullfile(out_session_dir, 'Wideband');
        out_events_dir = fullfile(out_session_dir, 'Events');
        
        % Validate Wideband directory existence
        if ~exist(in_wideband_dir, 'dir')
            fprintf('Skipping %s: Wideband directory not found.\n', session_name);
            continue;
        end
        
        % Mirror directory structure in raw_binary
        if ~exist(out_wideband_dir, 'dir')
            mkdir(out_wideband_dir);
        end
        if exist(in_events_dir, 'dir') && ~exist(out_events_dir, 'dir')
            mkdir(out_events_dir);
        end
        
        out_file = fullfile(out_wideband_dir, sprintf('%s_raw.bin', session_name));
        
        % Prevent overwriting existing binary files
        if exist(out_file, 'file')
            fprintf('Skipping %s: Binary file already exists.\n', session_name);
            continue;
        end
        
        % Retrieve total number of samples from the first channel
        file_1 = fullfile(in_wideband_dir, sprintf('%s_1.mat', session_name));
        if ~exist(file_1, 'file')
            fprintf('Skipping %s: %s_1.mat not found.\n', session_name, session_name);
            continue;
        end
        
        tmp = load(file_1);
        vars = fieldnames(tmp);
        data_1 = tmp.(vars{1}); 
        num_samples = length(data_1);
        
        % Initialize binary file writing
        fid = fopen(out_file, 'w');
        num_chunks = ceil(num_samples / chunk_size);
        
        fprintf('Processing %s: %d samples', session_name, num_samples); %[output:2919a3f6] %[output:7451c5da]

        for c = 1:num_chunks
            fprintf('Processing chunk: %03d/%03d', c, num_chunks); %[output:714ea252] %[output:9238df38] %[output:0b57b271] %[output:72149514] %[output:2fc31a59] %[output:845d7b2a] %[output:253c1a79] %[output:4e4d234b] %[output:4ce95ada] %[output:57f83660] %[output:9d5062fd] %[output:19e68c7b] %[output:6a27edf8] %[output:1f16fe4d] %[output:26525a4c] %[output:4f4cf400] %[output:8dcbfacd] %[output:9d9e5a26] %[output:6dad6c9c] %[output:5249a951] %[output:86582065] %[output:77ee9062] %[output:13f79ca1] %[output:47d08783] %[output:03f4328e] %[output:61163110] %[output:1e764376] %[output:07723b7d] %[output:2579cd51] %[output:087ed5a5] %[output:1604e072] %[output:99b53b5a] %[output:186f1f55] %[output:37b05b63] %[output:0f3a0ae9] %[output:3c76b986]
            start_idx = (c-1)*chunk_size + 1;
            end_idx = min(c*chunk_size, num_samples);
            current_chunk_samples = end_idx - start_idx + 1;
            
            % Preallocate chunk matrix [channels x samples]
            chunk_data = zeros(num_channels, current_chunk_samples, dtype);
            for ch = 1:num_channels
                fprintf('\b\b\b%03d', ch);                 %[output:4cc3186f] %[output:8b3089fe] %[output:0f97dcba] %[output:0b368bff] %[output:886d94bc] %[output:54cf5927] %[output:27b29cc2] %[output:4738e8c7] %[output:83031091] %[output:0d745acc] %[output:581423e8] %[output:53f6580e] %[output:18d54737] %[output:7515b687] %[output:598eb2f1] %[output:8ce4a8d7] %[output:2c3b30bd] %[output:0bff5e16] %[output:2c589488] %[output:7e1d629e] %[output:3c9d37ad] %[output:9e6471ca] %[output:3dd3698f] %[output:148f7bc5] %[output:3e7d0b14] %[output:8a26e6dc] %[output:90761764] %[output:3563ed41] %[output:1eb902ab] %[output:1d55054a] %[output:09380624] %[output:22e054c5] %[output:0f4fc67d] %[output:50783ab6] %[output:2d942411] %[output:91c593d1] %[output:066906ee]
                ch_file = fullfile(in_wideband_dir, sprintf('%s_%d.mat', session_name, ch));
                tmp = load(ch_file);
                vars2 = fieldnames(tmp);
                sig = single(tmp.(vars2{1}));
                chunk_data(ch, :) = sig(start_idx:end_idx);
            end
            % fwrite writes column by column, enabling multiplexed storage
            fwrite(fid, chunk_data, dtype);
        end
        
        fclose(fid);
        fprintf('Saved binary to: %s\n\n', out_file); %[output:3e447f67] %[output:6c13ce87]
    end
end %[output:group:6e441575]
%%
%[text] ### Events
% 1. Inspect subject directories
subjects = dir(matlab_input_dir);
subjects = subjects([subjects.isdir] & ~ismember({subjects.name}, {'.', '..'}));

for s = 1:length(subjects) %[output:group:195675b7]
    subj_name = subjects(s).name;
    subj_in_dir = fullfile(matlab_input_dir, subj_name);
    subj_out_dir = fullfile(binary_output_dir, subj_name);
    
    % 2. Inspect session directories
    sessions = dir(subj_in_dir);
    sessions = sessions([sessions.isdir] & ~ismember({sessions.name}, {'.', '..'}));
    
    for sess = 1:length(sessions)
        session_name = sessions(sess).name;
        
        in_events_dir = fullfile(subj_in_dir, session_name, 'Events');
        out_events_dir = fullfile(subj_out_dir, session_name, 'Events');
        
        % Check if Events folder exists in raw data
        if ~exist(in_events_dir, 'dir')
            continue;
        end
        
        % Mirror Events directory in raw_binary
        if ~exist(out_events_dir, 'dir')
            mkdir(out_events_dir);
        end
        
        % Process Walking/Steps Events (AllStepsTable)
        steps_file = fullfile(in_events_dir, sprintf('AllStepsTable_%s.mat', session_name));
        if exist(steps_file, 'file')
            tmp_steps = load(steps_file);
            vars = fieldnames(tmp_steps);
            steps_table = tmp_steps.(vars{1}); % Assuming the table is the first/only variable
            
            out_steps_csv = fullfile(out_events_dir, sprintf('%s_Steps.csv', session_name));
            writetable(steps_table, out_steps_csv);
            fprintf('Exported Steps table to: %s\n', out_steps_csv);
        end
        
        % Process Grasp Events (Grasp vectors)
        grasp_file = fullfile(in_events_dir, sprintf('%s_Grasp.mat', session_name));
        out_grasp_csv = fullfile(out_events_dir, sprintf('%s_Grasp.csv', session_name));
        if exist(grasp_file, 'file') && ~exist(out_grasp_csv, 'file')
            tmp_grasp = load(grasp_file);
            grasp_vars = fieldnames(tmp_grasp);
            
            % Initialize containers for the unified table
            all_times = [];
            all_hands = {};
            all_targets = {};
            all_event_types = {};
            
            for v = 1:length(grasp_vars)
                var_name = grasp_vars{v};
                timestamps = tmp_grasp.(var_name);
                
                % Parse metadata from variable name (e.g., Evt_Router_Grasp_to_eat_floor_L)
                parts = strsplit(var_name, '_');
                hand = parts{end};         % Extracts 'L' or 'R'
                target = parts{end-1};     % Extracts 'floor' or 'hook'
                
                num_events = length(timestamps);
                
                % Append to unified columns
                all_times = [all_times; timestamps];
                all_hands = [all_hands; repmat({hand}, num_events, 1)];
                all_targets = [all_targets; repmat({target}, num_events, 1)];
                all_event_types = [all_event_types; repmat({'Grasp_to_eat'}, num_events, 1)];
            end
            
            % Create the table and sort it chronologically
            grasp_table = table(all_times, all_event_types, all_targets, all_hands, ...
                'VariableNames', {'EventTime', 'EventType', 'Target', 'Hand'});
            grasp_table = sortrows(grasp_table, 'EventTime');
            
            writetable(grasp_table, out_grasp_csv);
            fprintf('Exported Unified Grasp table to: %s\n\n', out_grasp_csv); %[output:7524c0ea]
        elseif exist(out_grasp_csv, 'file')
            fprintf('Skipping Grasp export: %s already exists.\n\n', session_name);
        end
    end
end %[output:group:195675b7]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":21.3}
%---
%[output:2919a3f6]
%   data: {"dataType":"text","outputData":{"text":"Processing Router_20220211: 140788984 samples","truncated":false}}
%---
%[output:714ea252]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 001\/015","truncated":false}}
%---
%[output:4cc3186f]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:7451c5da]
%   data: {"dataType":"text","outputData":{"text":"Processing Wifi_20210618: 200509078 samples","truncated":false}}
%---
%[output:8b3089fe]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:9238df38]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 002\/015","truncated":false}}
%---
%[output:0f97dcba]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:0b57b271]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 003\/015","truncated":false}}
%---
%[output:0b368bff]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:72149514]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 004\/015","truncated":false}}
%---
%[output:886d94bc]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:2fc31a59]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 005\/015","truncated":false}}
%---
%[output:54cf5927]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:845d7b2a]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 006\/015","truncated":false}}
%---
%[output:27b29cc2]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:253c1a79]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 007\/015","truncated":false}}
%---
%[output:4738e8c7]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:83031091]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Some output might be missing due to a network interruption. To get the missing output, rerun the script."}}
%---
%[output:4e4d234b]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 008\/015","truncated":false}}
%---
%[output:0d745acc]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:4ce95ada]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 009\/015","truncated":false}}
%---
%[output:581423e8]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:57f83660]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 010\/015","truncated":false}}
%---
%[output:53f6580e]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:9d5062fd]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 011\/015","truncated":false}}
%---
%[output:18d54737]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:19e68c7b]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 012\/015","truncated":false}}
%---
%[output:7515b687]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:6a27edf8]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 013\/015","truncated":false}}
%---
%[output:598eb2f1]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:1f16fe4d]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 014\/015","truncated":false}}
%---
%[output:8ce4a8d7]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:3e447f67]
%   data: {"dataType":"text","outputData":{"text":"Saved binary to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Router\\Router_20220211\\Wideband\\Router_20220211_raw.bin\n\n","truncated":false}}
%---
%[output:26525a4c]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 015\/015","truncated":false}}
%---
%[output:4f4cf400]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 001\/021","truncated":false}}
%---
%[output:8dcbfacd]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 002\/021","truncated":false}}
%---
%[output:2c3b30bd]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:9d9e5a26]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 003\/021","truncated":false}}
%---
%[output:0bff5e16]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:6dad6c9c]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 004\/021","truncated":false}}
%---
%[output:2c589488]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:5249a951]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 005\/021","truncated":false}}
%---
%[output:7e1d629e]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:86582065]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 006\/021","truncated":false}}
%---
%[output:3c9d37ad]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:77ee9062]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 007\/021","truncated":false}}
%---
%[output:9e6471ca]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:13f79ca1]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 008\/021","truncated":false}}
%---
%[output:3dd3698f]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:47d08783]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 009\/021","truncated":false}}
%---
%[output:148f7bc5]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:03f4328e]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 010\/021","truncated":false}}
%---
%[output:3e7d0b14]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:61163110]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 011\/021","truncated":false}}
%---
%[output:8a26e6dc]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:1e764376]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 012\/021","truncated":false}}
%---
%[output:90761764]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:07723b7d]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 013\/021","truncated":false}}
%---
%[output:3563ed41]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:2579cd51]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 014\/021","truncated":false}}
%---
%[output:1eb902ab]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:087ed5a5]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 015\/021","truncated":false}}
%---
%[output:1d55054a]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:1604e072]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 016\/021","truncated":false}}
%---
%[output:09380624]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:99b53b5a]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 017\/021","truncated":false}}
%---
%[output:22e054c5]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:186f1f55]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 018\/021","truncated":false}}
%---
%[output:0f4fc67d]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:37b05b63]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 019\/021","truncated":false}}
%---
%[output:50783ab6]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:0f3a0ae9]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 020\/021","truncated":false}}
%---
%[output:2d942411]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:3c76b986]
%   data: {"dataType":"text","outputData":{"text":"Processing chunk: 021\/021","truncated":false}}
%---
%[output:91c593d1]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:066906ee]
%   data: {"dataType":"text","outputData":{"text":"128","truncated":false}}
%---
%[output:6c13ce87]
%   data: {"dataType":"text","outputData":{"text":"Saved binary to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Wifi\\Wifi_20210618\\Wideband\\Wifi_20210618_raw.bin\n\n","truncated":false}}
%---
%[output:7524c0ea]
%   data: {"dataType":"text","outputData":{"text":"Exported Unified Grasp table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Router\\Router_20220211\\Events\\Router_20220211_Grasp.csv\n\nExported Unified Grasp table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Wifi\\Wifi_20210618\\Events\\Wifi_20210618_Grasp.csv\n\n","truncated":false}}
%---
