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

for s = 1:length(subjects)
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
        
        fprintf('Processing %s: %d samples', session_name, num_samples);

        for c = 1:num_chunks
            fprintf('Processing chunk: %03d/%03d', c, num_chunks);
            start_idx = (c-1)*chunk_size + 1;
            end_idx = min(c*chunk_size, num_samples);
            current_chunk_samples = end_idx - start_idx + 1;
            
            % Preallocate chunk matrix [channels x samples]
            chunk_data = zeros(num_channels, current_chunk_samples, dtype);
            for ch = 1:num_channels
                fprintf('\b\b\b%03d', ch);                
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
        fprintf('Saved binary to: %s\n\n', out_file);
    end
end
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
        steps_file = fullfile(in_events_dir, sprintf('AllStepTable_%s.mat', session_name));
        if exist(steps_file, 'file')
            tmp_steps = load(steps_file);
            vars = fieldnames(tmp_steps);
            steps_table = tmp_steps.(vars{1}); % Assuming the table is the first/only variable
            
            out_steps_csv = fullfile(out_events_dir, sprintf('%s_Steps.csv', session_name));
            writetable(steps_table, out_steps_csv);
            fprintf('Exported Steps table to: %s\n', out_steps_csv); %[output:129f0258] %[output:2bcfe27a]
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
            fprintf('Exported Unified Grasp table to: %s\n\n', out_grasp_csv);
        elseif exist(out_grasp_csv, 'file')
            fprintf('Skipping Grasp export: %s already exists.\n\n', session_name); %[output:285ff0be] %[output:83ebf3b4]
        end
    end
end %[output:group:195675b7]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":21.3}
%---
%[output:129f0258]
%   data: {"dataType":"text","outputData":{"text":"Exported Steps table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Router\\Router_20220211\\Events\\Router_20220211_Steps.csv\n","truncated":false}}
%---
%[output:285ff0be]
%   data: {"dataType":"text","outputData":{"text":"Skipping Grasp export: Router_20220211 already exists.\n\n","truncated":false}}
%---
%[output:2bcfe27a]
%   data: {"dataType":"text","outputData":{"text":"Exported Steps table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Wifi\\Wifi_20210618\\Events\\Wifi_20210618_Steps.csv\n","truncated":false}}
%---
%[output:83ebf3b4]
%   data: {"dataType":"text","outputData":{"text":"Skipping Grasp export: Wifi_20210618 already exists.\n\n","truncated":false}}
%---
