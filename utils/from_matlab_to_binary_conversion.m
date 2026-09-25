%[text] ## **Matlab to binary converter**
%[text] This script converts high-density, single-channel `.mat` recordings (e.g., 32 kHz microelectrode data) into a unified, multiplexed binary file (`.bin`). The output format is explicitly designed for seamless integration with Python's **SpikeInterface** via the `read_binary` function.
%[text] **Core Functionalities**
%[text] - **Memory Efficiency**: Reads and writes one channel's full signal at a time, so at most one channel is ever held in memory.
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

% Session registry: single source of truth for the implanted hemisphere.
% No silent fallback if a session is missing or its hemisphere is unknown.
session_metadata_file = fullfile(base_dir, 'session_metadata.csv');
if ~exist(session_metadata_file, 'file')
    error('Session metadata registry not found at %s. Create it before running the converter.', session_metadata_file);
end
session_metadata = readtable(session_metadata_file, 'TextType', 'string');

% Processing parameters
num_channels = 128;
dtype = 'single';
%%
%[text] ### Signals conversion
% 1. Inspect subject directories
subjects = dir(matlab_input_dir);
subjects = subjects([subjects.isdir]);
subjects = subjects(~ismember({subjects.name}, {'.', '..'}));

for s = 1:length(subjects) %[output:group:76499ba2]
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
            fprintf('Skipping %s: Binary file already exists.\n', session_name); %[output:742af2bd]
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

        % Write each channel's full signal exactly once, instead of re-loading every
        % channel from disk on every chunk (matfile partial reads do not help here: the
        % source .mat files are not saved with -v7.3, so matfile silently loads the whole
        % variable into memory on first access anyway - verified against a synthetic
        % non-v7.3 file). fwrite's byte 'skip' places each channel directly into its
        % interleaved column position, so only one full [channels x samples] buffer (the
        % output file itself) exists on disk, never in memory.
        %
        % Two MATLAB I/O quirks the code below works around (verified empirically, not
        % documented clearly): (1) fseek to a position beyond the current end of a file
        % opened for writing fails silently (status -1, position unchanged), so the file
        % must already be its final size before any interior fseek; it is pre-sized here
        % with real zero-filled appends, in bounded chunks so this never holds the whole
        % file in memory. (2) fwrite(...,skip) inserts the skip BEFORE every value it
        % writes, including the first one in that call - so the first sample of each
        % channel is written on its own (lands exactly where fseek put it), and only the
        % remaining samples are written with skip.
        bytes_per_sample = 4; % 'single'
        total_bytes = num_channels * num_samples * bytes_per_sample;
        skip_bytes = (num_channels - 1) * bytes_per_sample;
        presize_chunk_bytes = 1e8;

        fid = fopen(out_file, 'w');
        bytes_written = 0;
        while bytes_written < total_bytes
            this_chunk = min(presize_chunk_bytes, total_bytes - bytes_written);
            fwrite(fid, zeros(1, this_chunk, 'uint8'));
            bytes_written = bytes_written + this_chunk;
        end

        fprintf('Processing %s: %d samples, %d channels\n', session_name, num_samples, num_channels);

        data_1 = single(data_1(:));
        fseek(fid, 0, 'bof');
        fwrite(fid, data_1(1), dtype);
        if numel(data_1) > 1
            fwrite(fid, data_1(2:end), dtype, skip_bytes);
        end
        clear tmp data_1;

        for ch = 2:num_channels
            fprintf('Processing channel: %03d/%03d\n', ch, num_channels);
            ch_file = fullfile(in_wideband_dir, sprintf('%s_%d.mat', session_name, ch));
            tmp = load(ch_file);
            vars2 = fieldnames(tmp);
            sig = single(tmp.(vars2{1}));
            sig = sig(:);
            if numel(sig) ~= num_samples
                error('Channel %d of %s has %d samples, expected %d (from channel 1).', ch, session_name, numel(sig), num_samples);
            end
            fseek(fid, (ch - 1) * bytes_per_sample, 'bof');
            fwrite(fid, sig(1), dtype);
            if numel(sig) > 1
                fwrite(fid, sig(2:end), dtype, skip_bytes);
            end
            clear tmp sig;
        end

        fclose(fid);
        fprintf('Saved binary to: %s\n\n', out_file);
    end
end %[output:group:76499ba2]
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

        % Resolve the implanted hemisphere from the registry; refuse to proceed
        % without it, since it determines the ipsi/contra -> L/R mapping below.
        session_row = session_metadata(session_metadata.Session == string(session_name), :);
        if height(session_row) == 0
            error('Session %s not found in session_metadata.csv. Add it to the registry before converting.', session_name);
        end
        hemisphere = upper(strtrim(char(session_row.Hemisphere(1))));
        if ~ismember(hemisphere, {'L', 'R'})
            error('Session %s has an unknown Hemisphere ("%s") in session_metadata.csv; expected L or R.', session_name, hemisphere);
        end
        if strcmp(hemisphere, 'L')
            contra_hand = 'R';
            ipsi_hand = 'L';
        else
            contra_hand = 'L';
            ipsi_hand = 'R';
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
            fprintf('Exported Steps table to: %s\n', out_steps_csv); %[output:3fcc73dd] %[output:54ce0dd9] %[output:48664d4e] %[output:38ace119]
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
                hand_raw = parts{end};         % Extracts 'L' or 'R'
                raw_target = parts{end-1};     % Extracts 'floor' or 'hook'

                % Standardize target naming to 'floor' or 'hook'; no unrecognized value
                % is allowed through, since downstream code assumes only these two.
                switch lower(raw_target)
                    case {'floor', 'food'}
                        target = 'floor';
                    case {'hook', 'foraging'}
                        target = 'hook';
                    otherwise
                        error('Unrecognized grasp target "%s" in variable %s of session %s.', raw_target, var_name, session_name);
                end

                % 'l'/'r' in the source variable name are already anatomical.
                % 'ipsi'/'contra' depend on the implanted hemisphere (session_metadata.csv).
                switch lower(hand_raw)
                    case 'l'
                        hand = 'L';
                    case 'r'
                        hand = 'R';
                    case 'ipsi'
                        hand = ipsi_hand;
                    case 'contra'
                        hand = contra_hand;
                    otherwise
                        error('Unrecognized grasp hand label "%s" in variable %s of session %s.', hand_raw, var_name, session_name);
                end
                
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
            fprintf('Exported Unified Grasp table to: %s\n\n', out_grasp_csv); %[output:25ce2300] %[output:97070ad0]
        elseif exist(out_grasp_csv, 'file')
            fprintf('Skipping Grasp export: %s already exists.\n\n', session_name); %[output:28fb0b4f] %[output:785b2f3e]
        end
    end
end %[output:group:195675b7]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":21.3}
%---
%[output:742af2bd]
%   data: {"dataType":"text","outputData":{"text":"Skipping Router_20211130: Binary file already exists.\nSkipping Router_20220211: Binary file already exists.\nSkipping Wifi_20210618: Binary file already exists.\nSkipping Wifi_20221020: Binary file already exists.\n","truncated":false}}
%---
%[output:3fcc73dd]
%   data: {"dataType":"text","outputData":{"text":"Exported Steps table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Router\\Router_20211130\\Events\\Router_20211130_Steps.csv\n","truncated":false}}
%---
%[output:25ce2300]
%   data: {"dataType":"text","outputData":{"text":"Exported Unified Grasp table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Router\\Router_20211130\\Events\\Router_20211130_Grasp.csv\n\n","truncated":false}}
%---
%[output:54ce0dd9]
%   data: {"dataType":"text","outputData":{"text":"Exported Steps table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Router\\Router_20220211\\Events\\Router_20220211_Steps.csv\n","truncated":false}}
%---
%[output:28fb0b4f]
%   data: {"dataType":"text","outputData":{"text":"Skipping Grasp export: Router_20220211 already exists.\n\n","truncated":false}}
%---
%[output:48664d4e]
%   data: {"dataType":"text","outputData":{"text":"Exported Steps table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Wifi\\Wifi_20210618\\Events\\Wifi_20210618_Steps.csv\n","truncated":false}}
%---
%[output:785b2f3e]
%   data: {"dataType":"text","outputData":{"text":"Skipping Grasp export: Wifi_20210618 already exists.\n\n","truncated":false}}
%---
%[output:38ace119]
%   data: {"dataType":"text","outputData":{"text":"Exported Steps table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Wifi\\Wifi_20221020\\Events\\Wifi_20221020_Steps.csv\n","truncated":false}}
%---
%[output:97070ad0]
%   data: {"dataType":"text","outputData":{"text":"Exported Unified Grasp table to: C:\\Users\\tommy\\OneDrive - Scuola Superiore Sant'Anna\\Monkeys Parma\\raw_binary\\Wifi\\Wifi_20221020\\Events\\Wifi_20221020_Grasp.csv\n\n","truncated":false}}
%---
