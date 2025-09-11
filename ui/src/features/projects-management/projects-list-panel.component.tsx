/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Dialog, DialogTrigger, Divider, Flex, PhotoPlaceholder, Text, View } from 'packages/ui';

import { ProjectsList } from './projects-list.component';

interface SelectedProjectProps {
    name: string;
}

const SelectedProjectButton = ({ name }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet>
            <View marginEnd={'size-100'}>{name}</View>
            <PhotoPlaceholder name={name} email='' height={'size-400'} width={'size-400'} />
        </ActionButton>
    );
};
export const ProjectsListPanel = ({ name: selectedProjectName }: SelectedProjectProps) => {
    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} />

            <Dialog width={'size-4600'}>
                <Flex
                    alignItems={'center'}
                    justifyContent={'center'}
                    width={'size-4600'}
                    marginTop={'size-200'}
                    gap={'size-200'}
                    direction={'column'}
                >
                    <PhotoPlaceholder name={selectedProjectName} email='' height={'size-1000'} width={'size-1000'} />
                    <Text>{selectedProjectName}</Text>
                    <Divider size='S' />
                    <ProjectsList />
                    <Divider size='S' />
                </Flex>
            </Dialog>
        </DialogTrigger>
    );
};
